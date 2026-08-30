import json
from pathlib import Path
from data_models import Function_definition, GenerationContext
from llm_sdk.llm_sdk import Small_LLM_Model
from generation_state import GenerationState, ParameterValueState, FunctionNameState, ParameterValueSeparatorState, ParametersState, ParameterNameState

def load_functions(path: Path) -> list[Function_definition]:
    with path.open() as file:
        functions = json.load(file)

    return [Function_definition(**function) for function in functions]

def get_tokens(
          model: Small_LLM_Model, text: str,) -> list[int]:
        """Encode text into token IDs."""

        return model.encode(
            text
        )[0].tolist()

# def build_function_prompt(user_prompt: str, functions: list[Function_definition]) -> str:
#     prompt = [f"{function.name}: {function.description}" for function in functions]
#     prompt.extend([
#                 "",
#                 "User prompt:",
#                 user_prompt,
#                 "",
#                 "Function:",
#             ])
#     return "\n".join(prompt)

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
) -> list[int]:

    generated: list[int] = []

    while state is not None:

        allowed_tokens = state.get_allowed()

        if not allowed_tokens:
            raise RuntimeError(
                f"No allowed tokens for "
                f"{type(state).__name__}"
            )

        logits = model.get_logits_from_input_ids(
            input_ids
        )

        highest_token_id = highest_allowed_tokens(
            logits,
            allowed_tokens,
        )

        input_ids.append(highest_token_id)
        generated.append(highest_token_id)

        state.add_token(highest_token_id)

        if state.should_transition(
            highest_token_id
        ):
            state = state.next_state()

            if state is None:
                break

    return generated

# def generate_tokens(
#     model: Small_LLM_Model,
#     input_ids: list[int],
#     state: GenerationState,
#     max_tokens: int = 150,
# ) -> list[int]:
#     generated_tokens: list[int] = []

#     for _ in range(max_tokens):
#         if state is None:
#             break

#         allowed_tokens = state.get_allowed()

#         if not allowed_tokens:
#             raise RuntimeError(
#                 f"No allowed tokens for state {type(state).__name__}"
#             )

#         logits = model.get_logits_from_input_ids(input_ids)
#         next_token_id = highest_allowed_tokens(logits, allowed_tokens)

#         input_ids.append(next_token_id)
#         generated_tokens.append(next_token_id)

#         # Update state history with generated token
#         state.add_token(next_token_id)

#         # Check for state transition
#         if state.should_transition(next_token_id):
#             state = state.next_state()

#     return generated_tokens


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
    # return list(vocabulary.keys())

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

def save_result(
    result: dict,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    if path.exists() and path.stat().st_size > 0:
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = []

    results.append(result)

    path.write_text(
        json.dumps(
            results,
            indent=4,
        ),
        encoding="utf-8",
    )

def main():
    model = Small_LLM_Model(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    functions = load_functions(
        Path("data/input/functions_definition.json")
    )
    function_tokens_sequence = tokenize_function_names(model,functions,)
    # user_prompt = "What is the sum of 25 and 89?"
    user_prompt = "What is the sum of -25 and 89?"
    # user_prompt = "What is the square root of 16?"
    # user_prompt = "Calculate the square root of 144"
    # user_prompt = "Reverse the string 'world'"
    # user_prompt = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    # user_prompt = "Greet shrek"

    # user_prompt = "Replace all vowels in 'Programming is fun' with asterisks"
     # user_prompt = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"

    function_prompt = build_function_prompt(
        user_prompt,
        functions,
    )

    print(function_prompt)

    input_ids = get_tokens(
        model,
        function_prompt,
    )

    context = GenerationContext(
        functions=functions,
        function_token_sequences=function_tokens_sequence,
        model=model,
        open_parameter_token = get_tokens(model, "(")[0],
        key_value_separator_token= get_tokens(model, ":")[0],
        parameter_separator_token= get_tokens(model, ",")[0],
        close_parameter_token= get_tokens(model, ")")[0],
        vocabulary=load_vocabulary(model),
    )

    state = FunctionNameState(context)

    generated = generate_tokens(
        model,
        input_ids,
        state,
    )

    print("\nGENERATED")
    print("---------")
    print(model.decode(generated))

    print("\nSELECTED FUNCTION")
    print("-----------------")
    print(context.selected_function)

    print("\nPARAMETER VALUES")
    print("----------------")
    print(context.parameter_values)

    result = build_result(
    user_prompt,
    context,
)

    save_result(
        result,
        Path(
            "data/output/function_calling_results.json"
        ),
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )
    
if __name__=="__main__":
    main()