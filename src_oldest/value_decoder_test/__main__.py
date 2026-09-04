import argparse
import json
from pathlib import Path

from .data_models import FunctionDefinition
from .llm_interface_old import LLMInterface


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate constrained function calls with an LLM."
    )

    parser.add_argument(
        "-f",
        "--functions-definition",
        type=Path,
        required=True,
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input JSON file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the output JSON file.",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model name.",
    )

    parser.add_argument(
        "-e",
        "--examine",
        action="store_true",
        help="Display generated token information.",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=30.0,
        help="Generation timeout in seconds.",
    )

    return parser.parse_args()


def load_functions(
    path: Path,
) -> list[FunctionDefinition]:
    """Load function definitions from JSON."""

    with path.open() as file:
        data = json.load(file)

    return [
        FunctionDefinition(**function)
        for function in data
    ]


def load_inputs(
    path: Path,
) -> list[str]:
    """Load prompts from JSON."""

    with path.open() as file:
        data = json.load(file)

    # Support a simple list:
    #
    # [
    #     "book a flight",
    #     "transfer 100 euros"
    # ]
    #
    if isinstance(data, list):

        return [
            str(item)
            for item in data
        ]

    # Support:
    #
    # {
    #     "inputs": [
    #         "book a flight"
    #     ]
    # }
    #
    if isinstance(data, dict):

        inputs = data.get("inputs")

        if isinstance(inputs, list):

            return [
                str(item)
                for item in inputs
            ]

    raise ValueError(
        "Input JSON must contain a list of prompts "
        "or an 'inputs' list."
    )


def main() -> None:
    """Run the application."""

    args = parse_args()

    functions = load_functions(
        args.functions_definition
    )

    prompts = load_inputs(
        args.input
    )

    interface = LLMInterface(
        args.model,
        functions,
    )

    for prompt in prompts:

        result = interface.process_prompt(
            prompt,
            timeout=args.timeout,
            examine=args.examine,
        )

        interface.dump(
            result,
            args.output,
        )


if __name__ == "__main__":
    main()