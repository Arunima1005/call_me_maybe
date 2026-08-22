from enum import Enum, auto
from typing import Any

from pydantic import BaseModel


class ConstraintState(Enum):
    START = auto()
    KEY = auto()
    VALUE = auto()
    AFTER_VALUE = auto()
    DONE = auto()


# class ParameterDefinition(BaseModel):
#     type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


class GrammarTokens(BaseModel):
    open_arguments: tuple[int, ...]
    key_value_separator: tuple[int, ...]
    separator: tuple[int, ...]
    close_arguments: tuple[int, ...]
class ParameterDecoder(BaseModel):
    function: FunctionDefinition
    grammar: GrammarTokens

    parameter_tokens: list[tuple[int, ...]]

    parameter_index: int = 0
    state: ConstraintState = ConstraintState.START

    generated_key_ids: list[int] = []
    generated_value_ids: list[int] = []

    def current_parameter(self):
        parameters = list(
            self.function.parameters.items()
        )

        if self.parameter_index >= len(parameters):
            return None

        return parameters[self.parameter_index]

    def current_parameter_tokens(self) -> tuple[int, ...]:
        return self.parameter_tokens[
            self.parameter_index
        ]

    def current_parameter_type(self) -> str:
        parameter = self.current_parameter()

        if parameter is None:
            raise IndexError("Parameter index out of range")

        return parameter[1]["type"]

    def next_parameter(self) -> None:
        self.parameter_index += 1
        self.generated_key_ids = []
        self.generated_value_ids = []