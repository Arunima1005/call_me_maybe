from llm_sdk.llm_sdk import Small_LLM_Model
import json
from model import FunctionDefinition


def load_functions_data(path: str) -> list[FunctionDefinition]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [FunctionDefinition.model_validate(item) for item in data]


def tokenize_function_names(
    model: Small_LLM_Model,
    functions_data: list[FunctionDefinition],
) -> list[list[int]]:
    functions_token_sequence = []
    for function in functions_data:
        functions_token_sequence.append((model.encode(function.name))[0].tolist())

    return functions_token_sequence

def tokenize_parameter_names(
    model: Small_LLM_Model,
    functions: FunctionDefinition):
    parameter_token = {}
    for param in functions.parameters:
        parameter_token[param] = (model.encode(param))[0].tolist()

    return parameter_token


def get_allowed_parameter_tokens(
    generated_param_ids: list[int],
    parameter_token_sequence: dict[str, list[int]],
) -> set[int]:
    allowed_tokens: set[int] = set()
    for sequence in parameter_token_sequence.values():
        if len(sequence) <= len(generated_param_ids):
            continue
        if sequence[:len(generated_param_ids)] == generated_param_ids:
            allowed_tokens.add(sequence[(len(generated_param_ids))])

    return allowed_tokens


def build_function_prompt(
    user_prompt: str, functions_data: list[FunctionDefinition]
) -> str:
    prompts = [
        "You are a function calling assistant.",
        "",
        "Available functions:",
        "",
    ]
    for function in functions_data:
        prompts.append(f"{function.name}: {function.description}")

    prompts.extend(
        [
            "",
            "User request:",
            user_prompt,
            "",
            "Function:",
        ]
    )

    return "\n".join(prompts)
def tokenize_text(model: Small_LLM_Model, text: str, ) -> list[int]:
    return model.encode(text)[0].tolist()

def build_parameter_prompt(
    user_prompt: str, function: FunctionDefinition
) -> str:
    prompts = [
        "You are a function calling assistant.",
        "",
        f"Function: {function.name}",
        f"Description: {function.description}",
        "",
        "Parameters:",
    ]
    for name, parameter in function.parameters.items():
        prompts.append(f"{name}: {parameter.type}")

    prompts.extend(
        [
            "",
            "User request:",
            user_prompt,
            "",
            "Arguments:",
        ]
    )

    return "\n".join(prompts)

def get_allowed_next_tokens(
    generated_fn_ids: list[int],
    functions_token_sequence: list[list[int]],
) -> set[int]:
    allowed_tokens: set[int] = set()
    for sequence in functions_token_sequence:
        if len(sequence) <= len(generated_fn_ids):
            continue
        if sequence[:len(generated_fn_ids)] == generated_fn_ids:
            allowed_tokens.add(sequence[(len(generated_fn_ids))])

    return allowed_tokens


def generate_function_name(
    model: Small_LLM_Model,
    input_ids: list[int],
    functions_token_sequence: list[list[int]],
) -> list[int]:
    generated_fn_ids: list[int] = []

    while True:
        allowed_tokens = get_allowed_next_tokens(
            generated_fn_ids, functions_token_sequence
        )
        if not allowed_tokens:
            break

        logits = model.get_logits_from_input_ids(input_ids)

        highest_token_id = None
        highest_logits = float("-inf")

        for token_id in allowed_tokens:
            if logits[token_id] > highest_logits:
                highest_logits = logits[token_id]
                highest_token_id = token_id

        generated_fn_ids.append(highest_token_id)
        input_ids.append(highest_token_id)

    return generated_fn_ids


def generate_parameter_name(
    model: Small_LLM_Model,
    input_ids: list[int],
    parameter_token_sequence: dict[str, list[int]],
) -> list[int]:
    generated_param_ids: list[int] = []

    while True:
        allowed_tokens = get_allowed_parameter_tokens(
            generated_param_ids, parameter_token_sequence
        )
        if not allowed_tokens:
            break

        logits = model.get_logits_from_input_ids(input_ids)

        highest_token_id = None
        highest_logits = float("-inf")

        for token_id in allowed_tokens:
            if logits[token_id] > highest_logits:
                highest_logits = logits[token_id]
                highest_token_id = token_id

        generated_param_ids.append(highest_token_id)
        input_ids.append(highest_token_id)

    return generated_param_ids

def find_function(functions_data: list[FunctionDefinition], selected_function_name: str,) -> FunctionDefinition:
    for function in functions_data:
        if function.name == selected_function_name:
            return function
    raise ValueError(f"Unknown function: {selected_function_name}")



def main() -> None:
    model = Small_LLM_Model()

    user_prompt = "What is the sum of 2 and 3?"

    functions_data = load_functions_data("data/input/functions_definition.json")

    prompt = build_function_prompt(
        user_prompt,
        functions_data,
    )

    print("MODEL PROMPT")
    print("------------")
    print(prompt)

    input_ids = model.encode(prompt)[0].tolist()

    functions_token_sequence = tokenize_function_names(
        model,
        functions_data,
    )
    print(functions_token_sequence)

    generated_ids = generate_function_name(
        model,
        input_ids,
        functions_token_sequence,
    )

    selected_function_name = model.decode(generated_ids)

    print("\nSELECTED FUNCTION")
    print("-----------------")
    print(selected_function_name)

    selected_function = find_function(
            functions_data,
            selected_function_name,
        )

    parameter_prompt = build_parameter_prompt(
    user_prompt,
    selected_function,
)

    print("\nPARAMETER PROMPT")
    print("----------------")
    print(parameter_prompt)


    parameter_input_ids = model.encode(parameter_prompt)[0].tolist()

    parameter_tokens_sequence = tokenize_parameter_names(model, selected_function)
    print("parameter_tokens_sequence")
    print(parameter_tokens_sequence)

    generate_parameter_id = generate_parameter_name(model, parameter_input_ids, parameter_tokens_sequence)
    print(f"generate_parameter_id = {generate_parameter_id}")

    selected_parameter = model.decode(generate_parameter_id)

    print("\nSELECTED PARAMETER")
    print("-----------------")
    print(selected_parameter)


if __name__ == "__main__":
    main()
