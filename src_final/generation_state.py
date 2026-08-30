from abc import ABC, abstractmethod
from data_models import Function_definition, GenerationContext
from src_final.parameter_values import ParameterValueFactory

class GenerationState(ABC):

    def __init__(self, context: GenerationContext):
        self.context = context
        self.tokens: list[int] = []

    @abstractmethod
    def get_allowed(self) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def next_state(self):
        raise NotImplementedError

    def add_token(self, token_id: int) -> None:
        self.tokens.append(token_id)

    def is_complete(self) -> bool:
        return False

    def should_transition(self, token_id: int) -> bool:
        return self.is_complete()

    def _valid_tokens(
        self,
        options: list[list[int]],
    ) -> set[int]:

        allowed = set()

        for option in options:

            if len(option) <= len(self.tokens):
                continue

            if option[:len(self.tokens)] == self.tokens:
                allowed.add(
                    option[len(self.tokens)]
                )

        return allowed

class FunctionNameState(GenerationState):

    def __init__(self, context: GenerationContext):
        super().__init__(context)

    def get_allowed(self) -> set[int]:
        return self._valid_tokens(
            self.context.function_token_sequences
        )

    def is_complete(self) -> bool:
        return self.tokens in self.context.function_token_sequences

    def next_state(self) -> GenerationState:

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

    def __init__(
        self,
        context: GenerationContext,
        # function: Function_definition,
    ):
        super().__init__(context)
        # self.function = function

    def get_allowed(self) -> set[int]:
        return {
            self.context.open_parameter_token
        }

    def is_complete(self) -> bool:
        return self.tokens == [
            self.context.open_parameter_token
        ]

    def next_state(self) -> GenerationState:
        return ParameterNameState(
            self.context,
        )

class ParameterNameState(GenerationState):

    def __init__(
        self,
        context: GenerationContext,
    ):
        super().__init__(context)

        parameter_names = list(
    self.context.selected_function.parameters
    )

        parameter_name = parameter_names[
            self.context.current_parameter_index
        ]

        self.parameter_names = [
            self.context.model.encode(parameter_name)[0].tolist()
        ]

    def get_allowed(self) -> set[int]:
        return self._valid_tokens(
            self.parameter_names
        )

    def is_complete(self) -> bool:
        return self.tokens in self.parameter_names

    def next_state(self) -> GenerationState:

        return ParameterValueSeparatorState(
            self.context
        )
    
class ParameterValueSeparatorState(GenerationState):

    def get_allowed(self) -> set[int]:
        return {
            self.context.key_value_separator_token
        }

    def is_complete(self) -> bool:
        return self.tokens == [
            self.context.key_value_separator_token
        ]

    def next_state(self) -> GenerationState:

        return ParameterValueState(
            self.context,
        )

# class ParameterValueState(GenerationState):

#     def __init__(self, context: GenerationContext):
#         super().__init__(context)

#         parameter_name = list(
#             self.context.selected_function.parameters
#         )[self.context.current_parameter_index]

#         parameter_type = (
#             self.context.selected_function.parameters[
#                 parameter_name
#             ]["type"]
#         )

#         self.value_rule = ParameterValueFactory.build(
#             parameter_type,
#             self.context.model,
#             self.context.vocabulary,
#         )

#     # def get_allowed(self) -> set[int]:

#     #     # Always allow tokens that can continue the value.
#     #     allowed = self.value_rule.get_allowed_tokens()

#     #     # Only allow , or ) AFTER we have a complete value.
#     #     if self.value_rule.can_finish():

#     #         if (
#     #             self.context.current_parameter_index + 1
#     #             < len(
#     #                 self.context.selected_function.parameters
#     #             )
#     #         ):
#     #             allowed.add(
#     #                 self.context.parameter_separator_token
#     #             )
#     #         else:
#     #             allowed.add(
#     #                 self.context.close_parameter_token
#     #             )

#     #     return allowed

#     def get_allowed(self) -> set[int]:
#         allowed = set(self.value_rule.get_allowed_tokens())

#         if self.value_rule.can_finish():
#             num_params = len(self.context.selected_function.parameters)
#             has_more_params = (self.context.current_parameter_index + 1) < num_params

#             if has_more_params:
#                 allowed.add(self.context.parameter_separator_token)
#             else:
#                 allowed.add(self.context.close_parameter_token)

#         # Safety Fallback: ensure execution doesn't stall if allowed set is empty
#         if not allowed:
#             num_params = len(self.context.selected_function.parameters)
#             if (self.context.current_parameter_index + 1) < num_params:
#                 allowed.add(self.context.parameter_separator_token)
#             else:
#                 allowed.add(self.context.close_parameter_token)

#         return allowed

    
#     def add_token(self, token_id: int) -> None:

#         # Structural token: do not add it to the value.
#         if self.is_termination_token(token_id):
#             return

#         self.value_rule.add_token(token_id)

#         if not self.value_rule.validate():
#             raise ValueError(
#                 f"Invalid parameter value: "
#                 f"{self.value_rule.get_value()}"
#             )

#     def is_complete(self) -> bool:
#         """
#         A value is complete when the value rule says
#         the current value is valid.

#         For example:
#             "2"  -> complete
#             "25" -> complete
#             "89" -> complete
#         """

#         return self.value_rule.can_finish()

#     def should_transition(self, token_id: int) -> bool:
#         """
#         We transition ONLY when the model generates
#         the structural token ',' or ')'.
#         """

#         return self.is_termination_token(token_id)

#     def is_termination_token(self, token_id: int) -> bool:

#         # Cannot terminate an empty/invalid value.
#         if not self.value_rule.can_finish():
#             return False

#         # More parameters -> comma terminates this value.
#         if (
#             self.context.current_parameter_index + 1
#             < len(
#                 self.context.selected_function.parameters
#             )
#         ):
#             return (
#                 token_id
#                 == self.context.parameter_separator_token
#             )

#         # Last parameter -> ')' terminates this value.
#         return (
#             token_id
#             == self.context.close_parameter_token
#         )

#     def get_value(self) -> str:
#         return self.value_rule.get_value()

#     def next_state(self) -> GenerationState:

#         parameter_name = list(
#             self.context.selected_function.parameters
#         )[self.context.current_parameter_index]

#         # Save the extracted value.
#         self.context.parameter_values[
#             parameter_name
#         ] = self.get_value()

#         # More parameters remain.
#         if (
#             self.context.current_parameter_index + 1
#             < len(
#                 self.context.selected_function.parameters
#             )
#         ):
#             self.context.current_parameter_index += 1

#             return ParameterNameState(
#                 self.context
#             )

#         # Last parameter is finished.
#         return None

class ParameterValueState(GenerationState):

    def __init__(self, context: GenerationContext):
        super().__init__(context)

        parameter_name = list(
            self.context.selected_function.parameters
        )[self.context.current_parameter_index]

        parameter_type = (
            self.context.selected_function.parameters[
                parameter_name
            ]["type"]
        )

        self.value_rule = ParameterValueFactory.build(
            parameter_type,
            self.context.model,
            self.context.vocabulary,
        )

    def get_allowed(self) -> set[int]:
        allowed = set(self.value_rule.get_allowed_tokens())

        if self.value_rule.can_finish():
            num_params = len(self.context.selected_function.parameters)
            has_more_params = (self.context.current_parameter_index + 1) < num_params

            if has_more_params:
                allowed.add(self.context.parameter_separator_token)
            else:
                allowed.add(self.context.close_parameter_token)

        # Safety Fallback: ensure execution doesn't stall if allowed set is empty
        if not allowed:
            num_params = len(self.context.selected_function.parameters)
            if (self.context.current_parameter_index + 1) < num_params:
                allowed.add(self.context.parameter_separator_token)
            else:
                allowed.add(self.context.close_parameter_token)

        return allowed

    def add_token(self, token_id: int) -> None:
        if self.is_termination_token(token_id):
            return

        self.value_rule.add_token(token_id)

        if not self.value_rule.validate():
            raise ValueError(
                f"Invalid parameter value: {self.value_rule.get_value()}"
            )

    def is_complete(self) -> bool:
        return self.value_rule.can_finish()

    def should_transition(self, token_id: int) -> bool:
        return self.is_termination_token(token_id)

    def is_termination_token(self, token_id: int) -> bool:
        if not self.value_rule.can_finish():
            return False

        if (self.context.current_parameter_index + 1) < len(self.context.selected_function.parameters):
            return token_id == self.context.parameter_separator_token

        return token_id == self.context.close_parameter_token

    def get_value(self) -> str:
        return self.value_rule.get_value()

    def next_state(self) -> GenerationState:
        parameter_name = list(
            self.context.selected_function.parameters
        )[self.context.current_parameter_index]

        self.context.parameter_values[parameter_name] = self.get_value()

        if (self.context.current_parameter_index + 1) < len(self.context.selected_function.parameters):
            self.context.current_parameter_index += 1
            return ParameterNameState(self.context)

        return None