from pathlib import Path
from argparse import Namespace

from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class ParameterDefinition(BaseModel):
    name: str
    type: str


class StateContext(BaseModel):
    """Information shared between decoding states."""

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

    model: object
    vocabulary: list[str]

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[
        str,
        str | int | float | bool | list | dict,
    ]


class Args(Namespace):
    functions_definition: list[FunctionDefinition]
    input: list[str]
    output: Path
    model: str
    dump: Path | None
    examine: bool
    timeout: float