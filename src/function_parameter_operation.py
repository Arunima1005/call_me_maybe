from llm_sdk.llm_sdk import Small_LLM_Model
import json,re
import numpy as np
from model import FunctionDefinition, ParameterDecoder, GrammarTokens, ConstraintState
from state_machine import next_state, get_allowed_tokens, get_allowed_sequence_tokens

NUMBER_PREFIX = re.compile(
    r"^-?(0|[1-9][0-9]*)(\.[0-9]*)?$"
)

INTEGER_PREFIX = re.compile(
    r"^-?(0|[1-9][0-9]*)$"
)

COMPLETE_NUMBER = re.compile(
    r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$"
)


def is_integer_prefix(value: str) -> bool:
    return bool(INTEGER_PREFIX.fullmatch(value))

def is_number_prefix(value: str) -> bool:
    return bool(NUMBER_PREFIX.fullmatch(value))

def is_complete_number(value: str) -> bool:
    return bool(COMPLETE_NUMBER.fullmatch(value))


def is_complete_integer(value: str) -> bool:
    return is_integer_prefix(value)

def is_value_prefix(
    value: str,
    parameter_type: str,
) -> bool:

    if parameter_type == "integer":
        return is_integer_prefix(value)

    if parameter_type == "number":
        return is_number_prefix(value)

    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )

def is_complete_value(
    value: str,
    parameter_type: str,
) -> bool:

    if parameter_type == "integer":
        return is_complete_integer(value)

    if parameter_type == "number":
        return is_complete_number(value)

    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )

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
        prompts.append(f"{name}: {list(parameter.values())[0]}")

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

        highest_token_id = highest_allowed_tokens(logits,allowed_tokens,)

        generated_fn_ids.append(highest_token_id)
        input_ids.append(highest_token_id)

    return generated_fn_ids

def find_function(functions_data: list[FunctionDefinition], selected_function_name: str,) -> FunctionDefinition:
    for function in functions_data:
        if function.name == selected_function_name:
            return function
    raise ValueError(f"Unknown function: {selected_function_name}")
def load_vocabulary(
    model: Small_LLM_Model,
) -> list[str]:

    with open(
        model.get_path_to_vocab_file(),
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    return list(vocabulary.keys())

def get_allowed_value_tokens(
        model: Small_LLM_Model,
        generate_value_ids: list[int],
        parameter_type: str,
        vocabulary: list[str],
) -> set[int]:
    allowed_tokens: set[int] = set()
    for token_id in range(len(vocabulary)):
        sample_tokens = generate_value_ids + [token_id]
        sample_value = model.decode(sample_tokens)
        if is_value_prefix(sample_value, parameter_type):
            allowed_tokens.add(token_id)
    return allowed_tokens
        
# def generate_parameter_value(model: Small_LLM_Model, parameter_input_ids: list[int], parameter_type: str, vocabulary: list[str], max_tokens: int = 5,):
#     generated_ids: list[int] = []
#     for _ in range(max_tokens):
#         allowed_tokens = get_allowed_value_tokens(model, generated_ids, parameter_type, vocabulary, )
#         if not allowed_tokens:
#             break
#         logits = model.get_logits_from_input_ids(parameter_input_ids)

#         top_token_ids = np.argsort(logits)[-10:][::-1]

#         print("\nTOP NEXT TOKENS")

#         for token_id in top_token_ids:
#             print(
#                 token_id,
#                 repr(vocabulary[token_id]),
#                 logits[token_id],
#             )

#         next_token = highest_allowed_tokens(logits, allowed_tokens,)
#         generated_ids.append(next_token)
#         parameter_input_ids.append(next_token)
#         # current_value = model.decode(
#         #             generated_ids
#         #         )
        
#         # print(
#         #     f"Value so far: {current_value!r}"
#         # )

#         # if is_complete_value(
#         #     current_value,
#         #     parameter_type,
#         # ):
#         #     print(
#         #         f"Complete value: {current_value!r}"
#         #     )
#     return model.decode(generated_ids)
def convert_parameter_value(
    value: str,
    parameter_type: str,
):

    if parameter_type == "integer":
        return int(value)

    if parameter_type == "number":
        return float(value)

    if parameter_type == "string":
        return value

    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )

def build_grammar_tokens(
    model: Small_LLM_Model,
) -> GrammarTokens:

    return GrammarTokens(
        open_arguments=tuple(
            model.encode("{")[0].tolist()
        ),
        key_value_separator=tuple(
            model.encode(":")[0].tolist()
        ),
        separator=tuple(
            model.encode(",")[0].tolist()
        ),
        close_arguments=tuple(
            model.encode("}")[0].tolist()
        ),
    )

def main() -> None:
    model = Small_LLM_Model()
    vocabulary = load_vocabulary(model)

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
    # print(functions_token_sequence)

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

    # print("\nDECODER")
    # print("-------")

    # while decoder.state != ConstraintState.DONE:

    #     print(
    #         "state:",
    #         decoder.state,
    #         "parameter:",
    #         decoder.current_parameter(),
    #     )

    #     decoder.next_state()

    # print("state:", decoder.state)

    parameter_prompt = build_parameter_prompt(
    user_prompt,
    selected_function,
)

    print("\nPARAMETER PROMPT")
    print("----------------")
    print(parameter_prompt)
    parameter_input_ids = model.encode(parameter_prompt)[0].tolist()
    grammar_tokens = build_grammar_tokens(model)
    decoder = ParameterDecoder(
    function=selected_function,
    grammar=grammar_tokens,
    parameter_tokens=[
        tuple(
            model.encode(name)[0].tolist()
        )
        for name in selected_function.parameters
    ],
)
    print("\nDEBUG KEY")
    print("---------")

    print("parameter_tokens:", decoder.parameter_tokens)
    print("current tokens:", decoder.current_parameter_tokens())
    print("generated:", decoder.generated_key_ids)

    print(
        "direct test:",
        get_allowed_sequence_tokens(
            decoder.generated_key_ids,
            [decoder.current_parameter_tokens()],
        ),
    )

    print("\nDECODER")
    print("-------")

    while decoder.state != ConstraintState.DONE:

        print(
            "state:",
            decoder.state,
            "parameter:",
            decoder.current_parameter(),
        )

        print("\nALLOWED TOKENS")
        print("-------------")

        for _ in range(4):

            print(
                "\nSTATE:",
                decoder.state,
                "PARAMETER:",
                decoder.current_parameter(),
            )

            allowed_tokens = get_allowed_tokens(
                decoder,
                model,
                vocabulary,
            )

            print("ALLOWED:", allowed_tokens)

            next_state(decoder)
    #     # for parameter_name, parameter_details in selected_function.parameters.items():
    #     # print(f"{parameter_name}: {list(parameter_type.values())[0]}")
    #     parameter_type = parameter_details["type"]
    #     print(f"{parameter_name}: {parameter_type}")

    #     parameter_input_ids = model.encode(parameter_prompt)[0].tolist()

    #     parameter_name_ids = model.encode(parameter_name)[0].tolist()

    #     parameter_input_ids.extend(parameter_name_ids)
    #     # parameter_type = list(parameter_type.values())[0]

    #     parameter_value = generate_parameter_value(model, parameter_input_ids, parameter_type, vocabulary)
    #     converted_parameter_value = (convert_parameter_value(parameter_value, parameter_type, ))
    #     parameters[parameter_name] = (converted_parameter_value)

    # print(f"SELECTED PARAMETER {parameters}")





if __name__ == "__main__":
    main()
