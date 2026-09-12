from abc import ABC, abstractmethod
from .data_models import Function_definition, GenerationContext
from .parameter_values import (
    ParameterValueRule,
    StringParameterRule,
    guess_regex_target,
    ParameterValueFactory,
    guess_replacement_from_prompt,
    assign_string_targets,
)


class GenerationState(ABC):
    """Base class for states in constrained function-call generation.

    Each state determines which tokens are allowed and when generation
    should transition to the next state.
    """

    def __init__(self, context: GenerationContext):
        """Initialize the state with the shared generation context."""
        self.context = context
        self.tokens: list[int] = []

    @abstractmethod
    def get_allowed(self) -> set[int]:
        """Return the token IDs allowed in the current state."""
        raise NotImplementedError

    @abstractmethod
    def next_state(self) -> "GenerationState | None":
        """Return the next generation state after this state completes."""
        raise NotImplementedError

    def add_token(self, token_id: int) -> None:
        """Add a generated token to the current state's token sequence."""
        self.tokens.append(token_id)

    def is_complete(self) -> bool:
        """Return whether the current state has completed its generation."""
        return False

    def should_transition(self, token_id: int) -> bool:
        """Return whether the generated token should
        trigger a state transition."""
        return self.is_complete()

    def _valid_tokens(
        self,
        options: list[list[int]],
    ) -> set[int]:
        """Return tokens that continue one of the given token sequences."""

        allowed = set()

        for option in options:

            if len(option) <= len(self.tokens):
                continue

            if option[: len(self.tokens)] == self.tokens:
                allowed.add(option[len(self.tokens)])

        return allowed

    def _require_selected_function(self) -> Function_definition:
        """Return the selected function,
        raising if none has been chosen yet."""
        if self.context.selected_function is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a selected function,"
                "but none is set"
            )
        return self.context.selected_function


class FunctionNameState(GenerationState):
    """Generates a valid function name from the available functions."""

    def __init__(self, context: GenerationContext):
        super().__init__(context)

    def get_allowed(self) -> set[int]:
        """Return tokens that can continue a valid function name."""
        return self._valid_tokens(self.context.function_token_sequences)

    def is_complete(self) -> bool:
        """Return whether a complete function name has been generated."""
        return self.tokens in self.context.function_token_sequences

    def next_state(self) -> "GenerationState | None":
        """Select the generated function and move to parameter generation."""

        for function, tokens in zip(
            self.context.functions,
            self.context.function_token_sequences,
        ):
            if tokens == self.tokens:
                self.context.selected_function = function
                return ParametersState(
                    self.context,
                )

        raise RuntimeError("Unknown function")


class ParametersState(GenerationState):
    """Generates the opening parenthesis before function parameters."""

    def get_allowed(self) -> set[int]:
        """Return the token representing the opening parenthesis."""
        return {self.context.open_parameter_token}

    def is_complete(self) -> bool:
        """Return whether the opening parenthesis has been generated."""
        return self.tokens == [self.context.open_parameter_token]

    def next_state(self) -> "GenerationState | None":
        """Move to generation of the first parameter name."""
        return ParameterNameState(
            self.context,
        )


class ParameterNameState(GenerationState):
    """Generates the name of the current function parameter."""

    def __init__(
        self,
        context: GenerationContext,
    ):
        """Initialize the state for the current parameter."""
        super().__init__(context)

        function = self.context.selected_function
        if function is None:
            raise RuntimeError("No Function Selected")

        parameter_names: list[str] = list(function.parameters)

        parameter_name: str = \
            parameter_names[self.context.current_parameter_index]

        self.parameter_names: list[list[int]] = [
            self.context.model.encode(parameter_name)[0].tolist()]

    def get_allowed(self) -> set[int]:
        """Return tokens that continue the current parameter name."""
        return self._valid_tokens(self.parameter_names)

    def is_complete(self) -> bool:
        """Return whether the complete parameter name has been generated."""
        return self.tokens in self.parameter_names

    def next_state(self) -> "GenerationState | None":
        """Move to generation of the parameter-value separator."""

        return ParameterValueSeparatorState(self.context)


class ParameterValueSeparatorState(GenerationState):
    """Generates the separator between a parameter name and its value."""

    def get_allowed(self) -> set[int]:
        """Return the token representing the key-value separator."""
        return {self.context.key_value_separator_token}

    def is_complete(self) -> bool:
        """Return whether the key-value separator has been generated."""
        return self.tokens == [self.context.key_value_separator_token]

    def next_state(self) -> "GenerationState | None":
        """Move to generation of the parameter value."""
        return ParameterValueState(
            self.context,
        )


class ParameterValueState(GenerationState):
    """Generates and validates the current parameter value.

    The value-generation rule depends on the parameter type. Values can be
    generated freely, generated using constrained rules, or reproduced
    exactly from a value extracted from the user prompt.
    """

    def __init__(self, context: GenerationContext):
        """Create the value rule for the current parameter."""
        super().__init__(context)

        function: Function_definition = self._require_selected_function()

        parameter_name: str = list(function.parameters)[
            self.context.current_parameter_index
        ]

        parameter_type: str = function.parameters[parameter_name]["type"]

        target = None
        if parameter_type == "string":
            string_targets = assign_string_targets(self.context)

            if (
                parameter_name.lower()
                in ParameterValueFactory.SYNTHESIZED_STRING_PARAMS
            ):
                target = guess_regex_target(self.context, string_targets)
            elif parameter_name.lower() == "replacement":
                target = guess_replacement_from_prompt(
                    self.context.user_prompt
                ) or string_targets.get(parameter_name)
            else:
                target = string_targets.get(parameter_name)

        self.value_rule: ParameterValueRule

        if parameter_type == "string" and target is not None:
            self.value_rule = StringParameterRule(
                self.context.model,
                self.context.vocabulary,
                target=target,
                mode="exact",
            )
        else:
            self.value_rule = ParameterValueFactory.build(
                parameter_type,
                self.context.model,
                self.context.vocabulary,
                target=target,
                parameter_name=parameter_name,
            )

    def get_allowed(self) -> set[int]:
        """Return valid value tokens and, when possible,
        a termination token."""

        if self.context.selected_function is None:
            raise RuntimeError(
                "get_allowed called with no function selected"
            )

        allowed = set(self.value_rule.get_allowed_tokens())

        if self.value_rule.can_finish():
            num_params: int = len(self.context.selected_function.parameters)
            has_more_params: bool = (
                self.context.current_parameter_index + 1) < num_params

            if has_more_params:
                allowed.add(self.context.parameter_separator_token)
            else:
                allowed.add(self.context.close_parameter_token)

        if not allowed:
            num_params = len(self.context.selected_function.parameters)
            if (self.context.current_parameter_index + 1) < num_params:
                allowed.add(self.context.parameter_separator_token)
            else:
                allowed.add(self.context.close_parameter_token)

        return allowed

    def add_token(self, token_id: int) -> None:
        """Add and validate a generated parameter-value token."""
        if self.should_transition(token_id):
            return

        self.value_rule.add_token(token_id)

        if not self.value_rule.validate():
            raise ValueError(
                f"Invalid parameter value: {self.value_rule.get_value()}")

    def should_transition(self, token_id: int) -> bool:
        """Return whether the generated token ends
        the current parameter value."""

        function = self._require_selected_function()

        if not self.value_rule.can_finish():
            return False

        if (self.context.current_parameter_index + 1) < len(
            function.parameters
        ):
            return token_id == self.context.parameter_separator_token

        return token_id == self.context.close_parameter_token

    def get_value(self) -> str:
        """Return the generated parameter value as text."""
        return self.value_rule.get_value()

    def next_state(self) -> "GenerationState | None":
        """Store the current value and move to the next parameter or finish."""
        function = self._require_selected_function()

        parameter_name: str = list(function.parameters)[
            self.context.current_parameter_index
        ]

        self.context.parameter_values[parameter_name] = self.get_value()

        if (self.context.current_parameter_index + 1) < len(
            function.parameters
        ):
            self.context.current_parameter_index += 1
            return ParameterNameState(self.context)

        return None
