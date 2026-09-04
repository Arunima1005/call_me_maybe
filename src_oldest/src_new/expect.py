from abc import ABC, abstractmethod

from .data_models import StateContext
from .decoder import ValueDecoder, ValueDecoderFactory


class Expect(ABC):
    """
    Base class for structural decoding states.
    """

    def __init__(self, context: StateContext):
        self.context = context

        # Tokens generated while matching a structural sequence.
        self.tokens: list[int] = []

    @abstractmethod
    def get_allowed(self) -> set[int]:
        """
        Return tokens allowed by this state.
        """

        raise NotImplementedError

    @abstractmethod
    def next_state(self) -> "Expect | None":
        """
        Move to the next structural state.
        """

        raise NotImplementedError


# ============================================================================
# Helper
# ============================================================================

class TokenSequenceExpect(Expect):
    """
    Base class for states that expect one specific token sequence.
    """

    def __init__(
        self,
        context: StateContext,
        expected: tuple[int, ...],
    ):
        super().__init__(context)

        self.expected = expected

    def get_allowed(self) -> set[int]:

        if not self.tokens:
            return {self.expected[0]}

        if len(self.tokens) >= len(self.expected):
            return set()

        for generated, expected in zip(
            self.tokens,
            self.expected,
        ):
            if generated != expected:
                return set()

        return {self.expected[len(self.tokens)]}


# ============================================================================
# Function
# ============================================================================

class ExpectFunction(Expect):
    """
    Expect the name of one of the available functions.
    """

    def get_allowed(self) -> set[int]:

        allowed: set[int] = set()

        for function_tokens in self.context.functions:

            if not self.tokens:
                allowed.add(function_tokens[0])
                continue

            if len(self.tokens) >= len(function_tokens):
                continue

            if all(
                generated == expected
                for generated, expected in zip(
                    self.tokens,
                    function_tokens,
                )
            ):
                allowed.add(function_tokens[len(self.tokens)])

        return allowed

    def next_state(self) -> Expect:

        selected_function = tuple(self.tokens)

        if selected_function not in self.context.functions:
            raise RuntimeError(
                "Generated function does not exist."
            )

        self.context.parameters = (
            self.context.functions[selected_function].copy()
        )

        return ExpectParameters(self.context)


# ============================================================================
# Parameters prefix
# ============================================================================

class ExpectParameters(TokenSequenceExpect):
    """
    Expect:

        parameters":{"
    """

    def __init__(self, context: StateContext):

        super().__init__(
            context,
            context.param_def,
        )

    def next_state(self) -> Expect:

        return ExpectKey(self.context)


# ============================================================================
# Parameter key
# ============================================================================

class ExpectKey(Expect):
    """
    Expect the next parameter name.
    """

    def get_allowed(self) -> set[int]:

        if not self.context.parameters:
            return set()

        parameter = self.context.parameters[0]

        expected = self.context.encode(
            f'"{parameter.name}"'
        )

        if not self.tokens:
            return {expected[0]}

        if len(self.tokens) >= len(expected):
            return set()

        if all(
            generated == expected_token
            for generated, expected_token in zip(
                self.tokens,
                expected,
            )
        ):
            return {expected[len(self.tokens)]}

        return set()

    def next_state(self) -> Expect:

        if not self.context.parameters:
            raise RuntimeError(
                "No parameter available."
            )

        self.context.current_parameter = (
            self.context.parameters.pop(0)
        )

        return ExpectKVSep(self.context)


# ============================================================================
# Key/value separator
# ============================================================================

class ExpectKVSep(TokenSequenceExpect):
    """
    Expect:

        :
    """

    def __init__(self, context: StateContext):

        super().__init__(
            context,
            context.kvsep,
        )

    def next_state(self) -> Expect:

        parameter = self.context.current_parameter

        if parameter is None:
            raise RuntimeError(
                "No current parameter."
            )

        return ExpectValue(
            self.context,
            parameter.type,
        )


# ============================================================================
# Value
# ============================================================================

class ExpectValue(Expect):
    """
    Generate the value using a type-specific ValueDecoder.
    """

    def __init__(
        self,
        context: StateContext,
        parameter_type: str,
    ):
        super().__init__(context)

        self.value_decoder: ValueDecoder = (
            ValueDecoderFactory.create(
                parameter_type=parameter_type,
                model=context.model,
                vocabulary=context.vocabulary,
            )
        )

    def get_allowed(self) -> set[int]:

        return self.value_decoder.get_allowed_tokens()

    def add_token(self, token_id: int) -> None:

        self.value_decoder.add_token(token_id)

    def can_finish(self) -> bool:

        return self.value_decoder.can_finish()

    def get_value(self) -> str:

        return self.value_decoder.get_value()

    def next_state(self) -> Expect:

        if not self.can_finish():
            raise RuntimeError(
                "Cannot leave value before it is complete."
            )

        if self.context.parameters:
            return ExpectSep(self.context)

        return ExpectEnd(self.context)


# ============================================================================
# Separator
# ============================================================================

class ExpectSep(TokenSequenceExpect):
    """
    Expect:

        ,
    """

    def __init__(self, context: StateContext):

        super().__init__(
            context,
            context.sep,
        )

    def next_state(self) -> Expect:

        return ExpectKey(self.context)


# ============================================================================
# End
# ============================================================================

class ExpectEnd(TokenSequenceExpect):
    """
    Expect:

        }}
    """

    def __init__(self, context: StateContext):

        super().__init__(
            context,
            context.end,
        )

    def next_state(self) -> None:

        return None