from argparse import Namespace
from pathlib import Path

from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    """Function definition following the provided JSON schema."""

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class ParameterDefinition(BaseModel):
    """Information needed to generate one parameter."""

    name: str
    type: str
    tokens: tuple[int, ...]


class StateContext(BaseModel):
    """Information shared between decoding states."""

    model: object
    vocabulary: list[str]

    functions: dict[
        tuple[int, ...],
        list[ParameterDefinition],
    ]

    parameters: list[ParameterDefinition] = Field(
        default_factory=list
    )

    current_parameter: ParameterDefinition | None = None

    param_def: tuple[int, ...]
    kvsep: tuple[int, ...]
    sep: tuple[int, ...]
    end: tuple[int, ...]

class FunctionCall(BaseModel):
    """Format that the output needs to follow."""

    prompt: str
    name: str
    parameters: dict[
        str,
        str | int | float | bool | list | dict,
    ]


class Args(Namespace):
    """Type specification for program arguments."""

    functions_definition: list[FunctionDefinition]
    input: list[str]
    output: Path
    model: str
    dump: Path | None
    examine: bool
    timeout: float