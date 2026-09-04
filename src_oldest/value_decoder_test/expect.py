"""States used for constrained function-call generation."""

from abc import ABC, abstractmethod

from .data_models import (
    ParameterDefinition,
    StateContext,
)
from .parameter_values import (
    ParameterValueFactory,
    ParameterValueRule,
    ValueCheck,
)


class Expect(ABC):
    """Base class for constrained decoding states."""

    def __init__(
        self,
        context: StateContext,
    ):
        self.context = context
        self.tokens: list[int] = []

    def _valid_tokens(
        self,
        options: tuple[tuple[int, ...], ...],
    ) -> set[int]:
        """Return tokens that continue an expected sequence."""

        if not options:
            return set()

        if not self.tokens:
            return {
                option[0]
                for option in options
                if option
            }

        allowed: set[int] = set()

        for option in options:

            if len(option) <= len(self.tokens):
                continue

            if all(
                expected == actual
                for expected, actual in zip(
                    option,
                    self.tokens,
                )
            ):
                allowed.add(
                    option[len(self.tokens)]
                )

        return allowed

    @abstractmethod
    def get_allowed(self) -> set[int]:
        """Return tokens allowed by this state."""

        raise NotImplementedError

    @abstractmethod
    def next_state(self) -> "Expect | None":
        """Return the next decoding state."""

        raise NotImplementedError


class ExpectFunction(Expect):
    """Generate the function name."""

    def get_allowed(self) -> set[int]:

        return self._valid_tokens(
            tuple(
                self.context.functions.keys()
            )
        )

    def next_state(self) -> Expect:

        function_tokens = tuple(
            self.tokens
        )

        if function_tokens not in self.context.functions:
            raise RuntimeError(
                "Generated function is not defined."
            )

        self.context.parameters = (
            self.context.functions[
                function_tokens
            ].copy()
        )

        return ExpectParameters(
            self.context
        )


class ExpectParameters(Expect):
    """Generate the opening parameter structure."""

    def get_allowed(self) -> set[int]:

        return self._valid_tokens(
            (
                self.context.param_def,
            )
        )

    def next_state(self) -> Expect:

        return ExpectKey(
            self.context
        )


class ExpectKey(Expect):
    """Generate a parameter name."""

    def get_allowed(self) -> set[int]:

        if not self.context.parameters:
            return set()

        return self._valid_tokens(
            (
                self.context.parameters[
                    0
                ].tokens,
            )
        )

    def next_state(self) -> Expect:

        if not self.context.parameters:
            raise RuntimeError(
                "No parameter is available."
            )

        parameter = (
            self.context.parameters[0]
        )

        return ExpectKVSep(
            self.context,
            parameter,
        )


class ExpectKVSep(Expect):
    """Generate the key/value separator."""

    def __init__(
        self,
        context: StateContext,
        parameter: ParameterDefinition,
    ):
        super().__init__(context)

        self.parameter = parameter

    def get_allowed(self) -> set[int]:

        return self._valid_tokens(
            (
                self.context.kvsep,
            )
        )

    def next_state(self) -> Expect:

        self.context.current_parameter = (
            self.parameter
        )

        return ExpectValue(
            self.context
        )

class ExpectValue(Expect):
    """
    Generate one parameter value.

    A value can be syntactically complete while the model
    may still want to continue generating it.

    Example:

        "2"

    is already a valid number, but it can still become:

        "265"

    Therefore completion is offered as a structural
    transition rather than being forced immediately.
    """

    def __init__(
        self,
        context: StateContext,
        model=None,
        vocabulary: list[str] | None = None,
        parameter_types: dict[tuple[int, ...], str] | None = None,
        parameter_type: str | None = None,
    ):
        super().__init__(context)

        # --------------------------------------------------------------
        # Unit tests can provide parameter_type directly.
        #
        # The real application gets it from current_parameter.
        # --------------------------------------------------------------

        if parameter_type is None:

            parameter = getattr(
                context,
                "current_parameter",
                None,
            )

            if parameter is None:
                raise RuntimeError(
                    "No current parameter."
                )

            parameter_type = parameter.type

        self.parameter_type = parameter_type

        # --------------------------------------------------------------
        # Use explicitly supplied model/vocabulary when provided.
        # Otherwise use StateContext.
        # --------------------------------------------------------------

        if model is None:
            model = context.model

        if vocabulary is None:
            vocabulary = context.vocabulary

        self.value_rule = (
            ParameterValueFactory.build(
                parameter_type=parameter_type,
                model=model,
                vocabulary=vocabulary,
            )
        )

    def get_allowed(self) -> set[int]:
        """
    Return tokens that can continue the value.
    """

        allowed = (
            self.value_rule
            .get_allowed_tokens()
        )

        # --------------------------------------------------------------
        # Before the value starts, don't allow the first token
        # of the structural separator/end sequence.
        # --------------------------------------------------------------

        if not self.value_rule.token_ids:

            termination_tokens = (
                self.get_termination_tokens()
            )

            allowed -= termination_tokens

        return allowed

    def check_token(
        self,
        token_id: int,
    ) -> ValueCheck:
        """
        Check a candidate value token without
        adding it.
        """

        return (
            self.value_rule
            .get_token_check(
                token_id
            )
        )

    def get_value(self) -> str:
        """Return the value generated so far."""

        return (
            self.value_rule
            .get_value()
        )

    def can_finish(self) -> bool:
        """
        Return whether the current value is
        syntactically complete.

        This does NOT transition to the next
        state by itself.
        """

        return (
            self.value_rule
            .can_finish()
        )

    def get_termination_tokens(
    self,
) -> set[int]:

        if not self.can_finish():
            return set()

        if self.is_last_parameter():
            sequence = self.context.end
        else:
            sequence = self.context.sep

        if not sequence:
            return set()

        return {
            sequence[0]
        }

    def is_termination_token(
        self,
        token_id: int,
    ) -> bool:
        """
        Return whether token_id is the structural
        token that terminates this value.
        """

        return (
            token_id
            in self.get_termination_tokens()
        )

    def add_token(
        self,
        token_id: int,
    ) -> None:
        """Add one actual value token."""

        self.value_rule.add_token(
            token_id
        )

    def next_state(self) -> Expect | None:
        """Move to the next state after a complete value."""

        if not self.can_finish():
            raise RuntimeError(
                "Parameter value is not complete."
            )

        self.context.parameters.pop(0)

        self.context.current_parameter = None

        if self.context.parameters:
            return ExpectSep(
                self.context
            )

        return ExpectEnd(
            self.context
        )

    
    def is_last_parameter(self) -> bool:
        return len(
            self.context.parameters
        ) == 1

class ExpectSep(Expect):
    """Generate the separator between parameters."""

    def get_allowed(self) -> set[int]:

        return self._valid_tokens(
            (
                self.context.sep,
            )
        )

    def next_state(self) -> Expect:

        return ExpectKey(
            self.context
        )


class ExpectEnd(Expect):
    """Generate the final closing structure."""

    def get_allowed(self) -> set[int]:

        return self._valid_tokens(
            (
                self.context.end,
            )
        )

    def next_state(self) -> None:
        """Finish generation."""

        return None