from model import ConstraintState, ParameterDecoder
from llm_sdk.llm_sdk import Small_LLM_Model
from value_constraints import get_allowed_value_tokens

def next_state(decoder: ParameterDecoder) -> None:

    if decoder.state == ConstraintState.START:
        decoder.state = ConstraintState.KEY
        return

    if decoder.state == ConstraintState.KEY:
        decoder.state = ConstraintState.VALUE
        return

    if decoder.state == ConstraintState.VALUE:
        decoder.state = ConstraintState.AFTER_VALUE
        return

    if decoder.state == ConstraintState.AFTER_VALUE:

        if (
            decoder.parameter_index + 1
            < len(decoder.function.parameters)
        ):
            decoder.next_parameter()
            decoder.state = ConstraintState.KEY
        else:
            decoder.state = ConstraintState.DONE

        return

    if decoder.state == ConstraintState.DONE:
        return

def get_allowed_tokens(
    decoder: ParameterDecoder,
    model: Small_LLM_Model,
    vocabulary: list[str],
) -> set[int]:

    if decoder.state == ConstraintState.START:
        return set(
            decoder.grammar.open_arguments
        )

    if decoder.state == ConstraintState.KEY:
        return get_allowed_sequence_tokens(
            decoder.generated_key_ids,
            [
                decoder.current_parameter_tokens()
            ],
        )

    if decoder.state == ConstraintState.VALUE:

        parameter_type = (
            decoder.current_parameter_type()
        )

        return get_allowed_value_tokens(
            model,
            decoder.generated_value_ids,
            parameter_type,
            vocabulary,
        )

    if decoder.state == ConstraintState.AFTER_VALUE:

        if (
            decoder.parameter_index + 1
            < len(decoder.function.parameters)
        ):
            return set(
                decoder.grammar.separator
            )

        return set(
            decoder.grammar.close_arguments
        )

    if decoder.state == ConstraintState.DONE:
        return set()

    raise ValueError(
        f"Unsupported state: {decoder.state}"
    )


def get_allowed_sequence_tokens(
    generated_ids: list[int],
    sequences: list[tuple[int, ...]],
) -> set[int]:

    allowed_tokens: set[int] = set()

    for sequence in sequences:

        if len(sequence) <= len(generated_ids):
            continue

        if sequence[:len(generated_ids)] == tuple(generated_ids):
            allowed_tokens.add(
                sequence[len(generated_ids)]
            )

    return allowed_tokens