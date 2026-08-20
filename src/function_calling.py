import json

import re

import numpy as np

from llm_sdk.llm_sdk import Small_LLM_Model

from model import FunctionDefinition


NUMBER_PREFIX = re.compile(
    r"^-?(0|[1-9][0-9]*)(\.[0-9]*)?$"
)

INTEGER_PREFIX = re.compile(
    r"^-?(0|[1-9][0-9]*)$"
)


def is_number_prefix(value: str) -> bool:
    return NUMBER_PREFIX.fullmatch(value) is not None


def is_integer_prefix(value: str) -> bool:
    return INTEGER_PREFIX.fullmatch(value) is not None


def is_string_prefix(value: str) -> bool:
    if not value.startswith('"'):
        return False

    escaped = False

    for char in value[1:]:
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True

    return True

def is_value_prefix(
    value: str,
    parameter_type: str,
) -> bool:

    if parameter_type == "number":
        return is_number_prefix(value)

    if parameter_type == "integer":
        return is_integer_prefix(value)

    if parameter_type == "string":
        return is_string_prefix(value)

    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )

def load_functions(path: str) -> list[FunctionDefinition]:

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        FunctionDefinition.model_validate(item)
        for item in data
    ]


def build_function_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    lines = [
        "You are a function calling assistant.",
        "",
        "Available functions:",
        "",
    ]

    for function in functions:
        lines.append(
            f"{function.name}: {function.description}"
        )

    lines.extend(
        [
            "",
            "User request:",
            user_prompt,
            "",
            "Function:",
        ]
    )

    return "\n".join(lines)

def build_parameter_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:
    lines = [
        "You are a function calling assistant.",
        "",
        f"Function: {function.name}",
        f"Description: {function.description}",
        "",
        "Parameters:",
    ]

    for name, parameter in function.parameters.items():
        lines.append(
        f"- {name}: {parameter.type}"
    )

    lines.extend(
        [
            "",
            "User request:",
            user_prompt,
            "",
            "Arguments:",
        ]
    )

    return "\n".join(lines)

def tokenize_function_names(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
) -> list[list[int]]:
    token_sequences = []

    for function in functions:
        token_ids = model.encode(
            function.name
        )[0].tolist()

        token_sequences.append(token_ids)

    return token_sequences


# def choose_best_allowed_token(
#     logits: np.ndarray,
#     allowed_tokens: set[int],
# ) -> int:

#     if not allowed_tokens:
#         raise ValueError("No valid next token")

#     allowed = np.array(
#         list(allowed_tokens),
#         dtype=np.int64,
#     )

#     best_index = np.argmax(
#         logits[allowed]
#     )

#     return int(allowed[best_index])

def mask_logits(
    logits: np.ndarray,
    allowed_tokens: set[int],
) -> np.ndarray:

    masked = np.full_like(
        logits,
        -np.inf,
        dtype=float,
    )

    for token_id in allowed_tokens:
        masked[token_id] = logits[token_id]

    return masked

def get_allowed_next_tokens(
    prefix: list[int],
    token_sequences: list[list[int]],
) -> set[int]:

    allowed_tokens: set[int] = set()

    for sequence in token_sequences:

        if len(sequence) <= len(prefix):
            continue

        if sequence[:len(prefix)] == prefix:
            allowed_tokens.add(
                sequence[len(prefix)]
            )

    return allowed_tokens


def choose_best_allowed_token(
    logits: list[float],
    allowed_tokens: set[int],
) -> int:
    if not allowed_tokens:
        raise ValueError("No valid next token")

    return max(
        allowed_tokens,
        key=lambda token_id: logits[token_id],
    )


def constrained_generate(
    model: Small_LLM_Model,
    input_ids: list[int],
    allowed_tokens_fn,
) -> list[int]:

    generated_ids: list[int] = []

    while True:

        allowed_tokens = allowed_tokens_fn(
            generated_ids
        )

        if not allowed_tokens:
            break

        logits = model.get_logits_from_input_ids(
            input_ids
        )

        next_token_id = choose_best_allowed_token(
            logits,
            allowed_tokens,
        )

        generated_ids.append(next_token_id)
        input_ids.append(next_token_id)

    return generated_ids

def generate_function_name(
    model,
    input_ids,
    token_sequences,
):
    return constrained_generate(
        model,
        input_ids,
        lambda prefix: get_allowed_next_tokens(
            prefix,
            token_sequences,
        ),
    )

def find_function(
    functions: list[FunctionDefinition],
    name: str,
) -> FunctionDefinition:
    for function in functions:
        if function.name == name:
            return function

    raise ValueError(
        f"Unknown function: {name}"
    )


def main() -> None:
    model = Small_LLM_Model()

    functions = load_functions(
        "data/input/functions_definition.json"
    )

    user_prompt = "What is the sum of 2 and 3?"

    prompt = build_function_prompt(
        user_prompt,
        functions,
    )

    print("MODEL PROMPT")
    print("------------")
    print(prompt)

    input_ids = model.encode(prompt)[0].tolist()

    token_sequences = tokenize_function_names(
        model,
        functions,
    )

    generated_ids = generate_function_name(
        model,
        input_ids,
        token_sequences,
    )

    selected_name = model.decode(generated_ids)

    print("\nSELECTED FUNCTION")
    print("-----------------")
    print(selected_name)

    selected_function = find_function(
        functions,
        selected_name,
    )

    print("\nPARAMETERS")
    print("----------")

    for name, parameter in selected_function.parameters.items():
        print(
        f"{name} -> {parameter.type}"
    )


if __name__ == "__main__":
    main()