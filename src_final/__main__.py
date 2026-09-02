import json
import time
from pathlib import Path
from data_models import Function_definition, GenerationContext
from llm_sdk.llm_sdk import Small_LLM_Model
from generation_state import GenerationState, ParameterValueState, FunctionNameState, ParameterValueSeparatorState, ParametersState, ParameterNameState

from parameter_values import extract_literal_candidates


RED = "\033[91m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"


def load_functions(path: Path) -> list[Function_definition]:
    if not path.is_file():
        print(RED + f"ERROR: Functions file not found: {path}" + RESET)
        # show_error_message(f"Functions file not found: {path}")
        raise SystemExit(1)
        
    with path.open() as file:
        functions = json.load(file)

    return [Function_definition(**function) for function in functions]

def load_user_prompts(path: Path) -> list[str]:
    if not path.is_file():
        print(RED + f"ERROR: User prompts file not found: {path}" + RESET)
        raise SystemExit(1)

    with path.open() as file:
        user_prompts = json.load(file)

    return [prompt["prompt"] for prompt in user_prompts]

def get_tokens(
          model: Small_LLM_Model, text: str,) -> list[int]:
        """Encode text into token IDs."""

        return model.encode(
            text
        )[0].tolist()

def build_function_prompt(
    user_prompt: str,
    functions: list[Function_definition],
) -> str:

    prompt = [
        "Select the function that best matches the user request.",
        "Then generate the function call using the values from the user request, preserving negative signs.",
        "Use the exact values mentioned by the user.",
        "",
    ]

    # prompt = [
    #     "Select the function that best matches the user request.",
    #     "Then generate the function call using the values from the user request, preserving negative signs.",
    #     "Use the exact values mentioned by the user.",
    #     "When a parameter needs a regex pattern, write a real regex (e.g. \\d+ for numbers, "
    #     "[aeiouAEIOU] for vowels, [^a-zA-Z0-9] for punctuation) — never copy a word from the prompt "
    #     "into a regex parameter.",
    #     "",
    #     "Example:",
    #     "User prompt: Replace all digits in 'a1b2' with X",
    #     "Function call: fn_substitute_string_with_regex(source_string:a1b2,regex:\\d+,replacement:X)",
    #     "",
    # ]

    for function in functions:
        prompt.append(
            f"{function.name}: {function.description}"
        )

    prompt.extend([
        "",
        "User prompt:",
        user_prompt,
        "",
        "Function call:",
    ])

    return "\n".join(prompt)

def tokenize_function_names(
    model: Small_LLM_Model,
    functions_data: list[Function_definition],
) -> list[list[int]]:
    functions_token_sequence = []
    for function in functions_data:
        functions_token_sequence.append((model.encode(function.name))[0].tolist())

    return functions_token_sequence


def find_function(functions_data: list[Function_definition], selected_function_name: str,) -> Function_definition:
    for function in functions_data:
        if function.name == selected_function_name:
            return function
    raise ValueError(f"Unknown function: {selected_function_name}")

def highest_allowed_tokens(
        logits: list[float],
        allowed_tokens: set[int],
) -> int:
    highest_token_id = None
    highest_logits = float("-inf")

    for token_id in allowed_tokens:
        if logits[token_id] > highest_logits:
            highest_logits = logits[token_id]
            highest_token_id = token_id
    return highest_token_id


def generate_tokens(
    model: Small_LLM_Model,
    input_ids: list[int],
    state: GenerationState,
    max_tokens: int = 200,
) -> list[int]:

    generated: list[int] = []

    while state is not None:

        if len(generated) >= max_tokens:
            raise RuntimeError(
                f"Generation exceeded {max_tokens} tokens without completing "
                f"(stuck in {type(state).__name__}); "
                f"partial output: {model.decode(generated)!r}"
            )

        allowed_tokens = state.get_allowed()

        if not allowed_tokens:
            raise RuntimeError(
                f"No allowed tokens for {type(state).__name__}"
            )

        logits = model.get_logits_from_input_ids(input_ids)
        highest_token_id = highest_allowed_tokens(logits, allowed_tokens)

        input_ids.append(highest_token_id)
        generated.append(highest_token_id)

        state.add_token(highest_token_id)

        if state.should_transition(highest_token_id):
            state = state.next_state()
            if state is None:
                break

    return generated


def load_vocabulary(
    model: Small_LLM_Model,
) -> list[str]:

    with open(
        model.get_path_to_vocab_file(),
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    result = [""] * (max(vocabulary.values()) + 1)

    for token_text, token_id in vocabulary.items():
        result[token_id] = token_text

    return result

def convert_value(
    value: str,
    parameter_type: str,
):
    if parameter_type == "number":
        return float(value)

    if parameter_type == "integer":
        return int(value)

    if parameter_type == "boolean":
        truthy_values = {"true", "1", "yes", "y", "on"}
        return value.strip().lower() in truthy_values

    if parameter_type == "string":
        return value

    if parameter_type == "array":
        return json.loads(value)

    if parameter_type == "object":
        return json.loads(value)

    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )

def build_result(
    prompt: str,
    context: GenerationContext,
) -> dict:

    function = context.selected_function

    parameters = {}

    for parameter_name, value in context.parameter_values.items():

        parameter_type = function.parameters[
            parameter_name
        ]["type"]

        parameters[parameter_name] = convert_value(
            value,
            parameter_type,
        )

    return {
        "prompt": prompt,
        "name": function.name,
        "parameters": parameters,
    }


def main():
    total_start_time = time.perf_counter()

    input_function_path = Path("data/input/functions_definition.json")
    
    input_user_prompt_path = Path("data/input/function_calling_tests.json")

    output_path = Path("data/output/function_calling_results.json")

    functions = load_functions(input_function_path)

    user_prompts = load_user_prompts(input_user_prompt_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = Small_LLM_Model(
            "Qwen/Qwen2.5-0.5B-Instruct"
        )

    function_tokens_sequence = tokenize_function_names(model,functions,)

    results = []
    total_prompts = len(user_prompts)
    width = 100
    print()
    print(CYAN + "=" * width + RESET)
    print(BOLD + "   CALL ME MAYBE   -     FUNCTION CALLING TEST RUN".center(width) + RESET)
    print(CYAN + "=" * width + RESET)
    print(f"{CYAN}Model   : {RESET}Qwen/Qwen2.5-0.5B-Instruct")
    print(f"{CYAN}Prompts : {RESET}{total_prompts}")
    print(CYAN+ "=" * width + RESET)

    for index, user_prompt in enumerate(user_prompts, start = 1):

        promt_start_time = time.perf_counter()

        print()
        print(CYAN + "┌" + "─" * (width - 2) + "┐" + RESET)
        print(CYAN+ f"│ PROMPT {index}/{total_prompts}".ljust(width - 1) + "│" + RESET)
        print(CYAN + "├" + "─" * (width - 2) + "┤" + RESET)
        print(CYAN + f"│ {user_prompt}".ljust((width - 1)) + "│" + RESET)
        print(CYAN + "└" + "─" * (width - 2) + "┘" + RESET)

        function_prompt = build_function_prompt(
            user_prompt,
            functions,
        )
        print()
        print(BOLD + CYAN+ f"System Prompt:" + RESET)
        print(BOLD + CYAN + "=" * width + RESET)
        for line in function_prompt.splitlines():
            print(BOLD + "    " + line + RESET)
        print(BOLD + CYAN + "=" * width + RESET)
        print()
        input_ids = get_tokens(
            model,
            function_prompt,
        )

        literal_candidates = extract_literal_candidates(user_prompt)

        context = GenerationContext(
            functions=functions,
            function_token_sequences=function_tokens_sequence,
            model=model,
            open_parameter_token = get_tokens(model, "(")[0],
            key_value_separator_token= get_tokens(model, ":")[0],
            parameter_separator_token= get_tokens(model, ",")[0],
            close_parameter_token= get_tokens(model, ")")[0],
            vocabulary=load_vocabulary(model),
            literal_candidates=literal_candidates,
            user_prompt=user_prompt,   
        )

        state = FunctionNameState(context)

        generated = generate_tokens(
            model,
            input_ids,
            state,
        )

        result = build_result(
        user_prompt,
        context,)

        results.append(result)
        output_path.write_text(
            json.dumps(results, indent=4),
            encoding="utf-8",
        )

        prompt_execution_time = (time.perf_counter() - promt_start_time)

        print()
        print(GREEN + "GENERATED" + RESET)
        print(GREEN + "  ✓ FUNCTION" + RESET)
        print(f"    {result['name']}")

        print()
        print(GREEN + "  ✓ PARAMETERS" + RESET)

        if result["parameters"]:
            for name, value in result["parameters"].items():
                print(f"    {name} = {value!r}")
        else:
            print("    None")

        print()
        print(YELLOW + f"  Execution time: {prompt_execution_time:.3f} seconds" + RESET)

    total_execution_time = (
        time.perf_counter() - total_start_time
    )

    print()
    print(GREEN + "=" * width + RESET)
    print(BOLD + "                    TEST RUN COMPLETE" + RESET)
    print(GREEN + "=" * width + RESET)
    print(f"  Prompts executed : {total_prompts}")
    print(f"  Results saved    : {len(results)}")
    print(f"  Total time       : {total_execution_time:.3f} seconds")
    print(f"  Output file      : {output_path}")
    print(GREEN + "=" * width + RESET)

    
if __name__=="__main__":
    main()