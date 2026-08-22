import re

from llm_sdk.llm_sdk import Small_LLM_Model


NUMBER_PREFIX = re.compile(
    r"^-?(0|[1-9][0-9]*)(\.[0-9]*)?$"
)

INTEGER_PREFIX = re.compile(
    r"^-?(0|[1-9][0-9]*)$"
)

COMPLETE_NUMBER = re.compile(
    r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$"
)


def is_number_prefix(value: str) -> bool:
    return bool(
        NUMBER_PREFIX.fullmatch(value)
    )


def is_integer_prefix(value: str) -> bool:
    return bool(
        INTEGER_PREFIX.fullmatch(value)
    )


def is_complete_number(value: str) -> bool:
    return bool(
        COMPLETE_NUMBER.fullmatch(value)
    )


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
