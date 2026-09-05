import json
from typing import cast
import argparse
import time
from typing import Any
from pathlib import Path
from pydantic import ValidationError
from .data_models import Function_definition, GenerationContext
from llm_sdk import Small_LLM_Model
from .generation_state import (
    GenerationState,
    FunctionNameState,
)

from .parameter_values import extract_literal_candidates

RED = "\033[91m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"


def load_functions(path: Path) -> list[Function_definition]:
    """Load function definitions from a JSON file."""
    if not path.is_file():
        print(RED + f"ERROR: Functions file not found: {path}" + RESET)
        raise SystemExit(1)

    with path.open() as file:
        try:
            functions = json.load(file)
        except json.JSONDecodeError:
            print(
                RED
                + f"ERROR: Invalid JSON in functions file: {path}"
                + RESET
            )
            raise SystemExit(1)

    try:
        function_definitions = [
            Function_definition(**function)
            for function in functions
        ]
    except ValidationError:
        print(
            RED
            + f"ERROR: Invalid function definition in file: {path}"
            + RESET
        )
        raise SystemExit(1)

    if not function_definitions:
        print(
            RED
            + f"ERROR: No function definitions found in file: {path}"
            + RESET
        )
        raise SystemExit(1)

    return function_definitions


def load_user_prompts(path: Path) -> list[str]:
    """Load user prompts from a JSON file."""
    if not path.is_file():
        print(RED + f"ERROR: User prompts file not found: {path}" + RESET)
        raise SystemExit(1)

    with path.open() as file:
        try:
            user_prompts = json.load(file)
        except json.JSONDecodeError:
            print(
                RED
                + f"ERROR: Invalid JSON in user prompts file: {path}"
                + RESET
            )
            raise SystemExit(1)
    if not user_prompts:
        print(
            RED
            + f"ERROR: No user prompts found in file: {path}"
            + RESET
        )
        raise SystemExit(1)

    try:
        prompts = [prompt["prompt"] for prompt in user_prompts]
    except (KeyError, TypeError):
        print(
            RED
            + f"ERROR: Invalid user prompt definition in file: {path}"
            + RESET
        )
        raise SystemExit(1)

    return prompts


def get_tokens(
    model: Small_LLM_Model,
    text: str,
) -> list[int]:
    """Encode text into token IDs."""
    return cast(list[int], model.encode(text)[0].tolist())


def build_function_prompt(
    user_prompt: str,
    functions: list[Function_definition],
) -> str:
    """Build the prompt used to generate a function call."""

    prompt = [
        "Select the function that best matches the user request.",
        "Then generate the function call "
        "using the values from the user request,"
        "Use the exact values mentioned by the user.",
        "",
    ]

    for function in functions:
        prompt.append(f"{function.name}: {function.description}")

    prompt.extend(
        [
            "",
            "User prompt:",
            user_prompt,
            "",
            "Function call:",
        ]
    )
    return "\n".join(prompt)


def tokenize_function_names(
    model: Small_LLM_Model,
    functions_data: list[Function_definition],
) -> list[list[int]]:
    """Tokenize all available function names."""
    functions_token_sequence = []
    for function in functions_data:
        functions_token_sequence.append(
            (model.encode(function.name))[0].tolist())

    return functions_token_sequence


def find_function(
    functions_data: list[Function_definition],
    selected_function_name: str,
) -> Function_definition:
    """Find a function definition by name."""
    for function in functions_data:
        if function.name == selected_function_name:
            return function
    raise ValueError(f"Unknown function: {selected_function_name}")


def highest_allowed_tokens(
    logits: list[float],
    allowed_tokens: set[int],
) -> int:
    """Return the allowed token with the highest logit."""
    highest_token_id: int | None = None
    highest_logits = float("-inf")

    for token_id in allowed_tokens:
        if logits[token_id] > highest_logits:
            highest_logits = logits[token_id]
            highest_token_id = token_id

    if highest_token_id is None:
        raise RuntimeError(
            "highest_allowed_tokens called with an empty allowed_tokens set"
        )
    return highest_token_id


def generate_tokens(
    model: Small_LLM_Model,
    input_ids: list[int],
    state: GenerationState | None,
    max_tokens: int = 200,
) -> list[int]:
    """Generate tokens while following the current generation state."""

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
            raise RuntimeError(f"No allowed tokens for {type(state).__name__}")

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
    """Load the model vocabulary indexed by token ID."""

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
) -> Any:
    # -> float | int | bool | str | list[Any] | dict[str, Any]:
    """Convert a generated string to the requested parameter type."""
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

    raise ValueError(f"Unsupported parameter type: {parameter_type}")


def build_result(
    prompt: str,
    context: GenerationContext,
) -> dict[str, Any]:
    """Build the final function-call result."""

    if context.selected_function is None:
        raise RuntimeError(
            "build_result called with no function selected"
        )

    function: Function_definition = context.selected_function

    parameters: dict[str, Any] = {}

    for parameter_name, value in context.parameter_values.items():
        parameter_type: str = function.parameters[parameter_name]["type"]
        parameters[parameter_name] = convert_value(
            value,
            parameter_type,
        )

    return {
        "prompt": prompt,
        "name": function.name,
        "parameters": parameters,
    }


def main() -> None:
    """Run the function-calling test application."""
    total_start_time = time.perf_counter()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--functions",
        default="data/input/functions_definition.json",
        help="Path to functions definition JSON",
    )

    parser.add_argument(
        "-i",
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to user prompts JSON",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to output JSON",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name",
    )

    args = parser.parse_args()

    input_function_path = Path(args.functions)
    input_user_prompt_path = Path(args.input)
    output_path = Path(args.output)
    functions = load_functions(input_function_path)
    user_prompts = load_user_prompts(input_user_prompt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = Small_LLM_Model(args.model)

    function_tokens_sequence = tokenize_function_names(
        model,
        functions,
    )

    results = []
    total_prompts = len(user_prompts)
    width = 100
    print()
    print(CYAN + "=" * width + RESET)
    print(
        BOLD
        + "   CALL ME MAYBE   -     FUNCTION CALLING TEST RUN".center(width)
        + RESET
    )
    print(CYAN + "=" * width + RESET)
    print(f"{CYAN}Model   : {RESET}{args.model}")
    print(f"{CYAN}Prompts : {RESET}{total_prompts}")
    print(CYAN + "=" * width + RESET)

    for index, user_prompt in enumerate(user_prompts, start=1):

        prompt_start_time = time.perf_counter()

        print()
        print(CYAN + "┌" + "─" * (width - 2) + "┐" + RESET)
        print(
            CYAN + "│ PROMPT "
            f"{index} / {total_prompts}".ljust(width - 1)
            + "│" + RESET
        )
        print(CYAN + "├" + "─" * (width - 2) + "┤" + RESET)
        print(CYAN + f"│ {user_prompt}".ljust((width - 1)) + "│"
              + RESET)
        print(CYAN + "└" + "─" * (width - 2) + "┘" + RESET)

        try:

            function_prompt = build_function_prompt(
                user_prompt,
                functions,
            )
            # print()
            # print(BOLD + CYAN+ f"System Prompt:" + RESET)
            # print(BOLD + CYAN + "=" * width + RESET)
            # for line in function_prompt.splitlines():
            #     print(BOLD + "    " + line + RESET)
            # print(BOLD + CYAN + "=" * width + RESET)
            # print()
            input_ids = get_tokens(
                model,
                function_prompt,
            )

            literal_candidates = extract_literal_candidates(user_prompt)

            context = GenerationContext(
                functions=functions,
                function_token_sequences=function_tokens_sequence,
                model=model,
                open_parameter_token=get_tokens(model, "(")[0],
                key_value_separator_token=get_tokens(model, ":")[0],
                parameter_separator_token=get_tokens(model, ",")[0],
                close_parameter_token=get_tokens(model, ")")[0],
                vocabulary=load_vocabulary(model),
                literal_candidates=literal_candidates,
                user_prompt=user_prompt,
            )

            state: GenerationState | None = FunctionNameState(context)

            generate_tokens(
                model,
                input_ids,
                state,
            )

            result = build_result(
                user_prompt,
                context,
            )
        except (RuntimeError, ValidationError, ValueError) as error:
            print()
            print(RED + BOLD + "  FAILED" + RESET)
            print(f"    {error}")
            prompt_execution_time = time.perf_counter() - prompt_start_time
            print()
            print(
                YELLOW + "  Execution time: " +
                f"{prompt_execution_time:.3f} seconds" + RESET
            )
            continue

        results.append(result)
        output_path.write_text(
            json.dumps(results, indent=4),
            encoding="utf-8",
        )

        prompt_execution_time = \
            time.perf_counter() - prompt_start_time
        minutes = int(prompt_execution_time // 60)
        seconds = int(prompt_execution_time % 60)

        print()
        print(GREEN + BOLD + "  FUNCTION CALL - OUTPUT" + RESET)
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
        print(
            YELLOW
            + "  Execution time: "
            + f"{minutes} minute{'s' if minutes != 1 else ''} "
            + f"and {seconds} second{'s' if seconds != 1 else ''}"
            + RESET
        )
    total_execution_time = \
        time.perf_counter() - total_start_time
    minutes = int(total_execution_time // 60)
    seconds = int(total_execution_time % 60)

    print()
    print(GREEN + "=" * width + RESET)
    print(BOLD + "                    TEST RUN COMPLETE" + RESET)
    print(GREEN + "=" * width + RESET)
    print(f"  Prompts executed : {total_prompts}")
    print(f"  Results saved    : {len(results)}")
    print(
        f"  Total time       : "
        f"{minutes} minute{'s' if minutes != 0 else ''} "
        f"and {seconds} second{'s' if seconds != 1 else ''}"
    )
    print(f"  Output file      : {output_path}")
    print(GREEN + "=" * width + RESET)


if __name__ == "__main__":
    main()
