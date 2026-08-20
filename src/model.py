from enum import Enum, auto
from typing import Any

from pydantic import BaseModel


class ConstraintState(Enum):
    START = auto()
    KEY = auto()
    VALUE = auto()
    AFTER_VALUE = auto()
    DONE = auto()


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
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
    parameter_index: int = 0
    state: ConstraintState = ConstraintState.START
    generated: str = ""

    def current_parameter_name(self) -> str:
        names = list(self.function.parameters)

        if self.parameter_index >= len(names):
            raise IndexError("No more parameters")

        return names[self.parameter_index]

    def current_parameter(self) -> ParameterDefinition:
        name = self.current_parameter_name()
        return self.function.parameters[name]

    def has_more_parameters(self) -> bool:
        return (
            self.parameter_index
            < len(self.function.parameters)
        )

# from dataclasses import dataclass
# from typing import Any
# from enum import Enum, auto

# class ConstraintState(Enum):
#     START = auto()
#     KEY = auto()
#     VALUE = auto()
#     AFTER_VALUE = auto()
#     DONE = auto()

# @dataclass(frozen=True)
# class Parameter:
#     name: str
#     type: str

# @dataclass(frozen=True)
# class FunctionDefinition:
#     name: str
#     description: str
#     parameters: list[Parameter]

# @dataclass(frozen=True)
# class FunctionCall:
#     name: str
#     parameters: dict[str, Any]

# @dataclass(slots=True)
# class ParameterDecoder:
#     function: FunctionDefinition
#     parameter_index: int = 0
#     state: ConstraintState = ConstraintState.START
