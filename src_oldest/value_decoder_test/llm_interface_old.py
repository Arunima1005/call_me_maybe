import json
import time
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model

from .data_models import (
    FunctionCall,
    FunctionDefinition,
    ParameterDefinition,
    StateContext,
)
from .expect import (
    Expect,
    ExpectFunction,
    ExpectValue,
)


class LLMInterface:
    """Interface used to interact with the LLM."""

    def __init__(
        self,
        model_name: str,
        defs: list[FunctionDefinition],
    ):
        print(
            f"Current model: {model_name}\n"
        )

        self.model = Small_LLM_Model(
            model_name
        )

        self.vocab: list[str] = list(
            json.loads(
                Path(
                    self.model.get_path_to_vocab_file()
                ).read_text()
            ).keys()
        )

        # --------------------------------------------------------------
        # Context used for function generation.
        # --------------------------------------------------------------

        self.context = self.get_tokens(
            f"""Available functions:
{'\n'.join(
    [
        ': '.join(
            [
                function.name,
                function.description,
            ]
        )
        for function in defs
    ]
)}"""
        )

        if defs:
            print(
                "Context\n"
                f"{self.model.decode(self.context)}"
            )

        # --------------------------------------------------------------
        # Build function map.
        #
        # function token IDs
        #        ↓
        # list[ParameterDefinition]
        #
        # ParameterDefinition contains:
        #     name
        #     type
        #     tokens
        # --------------------------------------------------------------

        functions: dict[
            tuple[int, ...],
            list[ParameterDefinition],
        ] = {}

        for function in defs:

            function_tokens = tuple(
                self.get_tokens(
                    function.name
                )
                + self.get_tokens(
                    '","'
                )
            )

            parameters: list[
                ParameterDefinition
            ] = []

            for (
                parameter_name,
                parameter_definition,
            ) in function.parameters.items():

                parameter_tokens = tuple(
                    self.get_tokens(
                        parameter_name
                    )
                )

                parameters.append(
                    ParameterDefinition(
                        name=parameter_name,
                        type=parameter_definition[
                            "type"
                        ],
                        tokens=parameter_tokens,
                    )
                )

            functions[
                function_tokens
            ] = parameters

        # --------------------------------------------------------------
        # Shared decoding context.
        # --------------------------------------------------------------

        self.state_context = StateContext(
            model=self.model,
            vocabulary=self.vocab,
            functions=functions,

            param_def=tuple(
                self.get_tokens(
                    'parameters":{"'
                )
            ),

            kvsep=tuple(
                self.get_tokens(
                    '":"'
                )
            ),

            sep=tuple(
                self.get_tokens(
                    '","'
                )
            ),

            end=tuple(
                self.get_tokens(
                    '"}}\n'
                )
            ),
        )

        # --------------------------------------------------------------
        # Conversion functions used when dumping.
        # --------------------------------------------------------------

        def convert_array(
            value: str,
        ) -> list[Any]:
            """Convert a generated array value."""

            value = value.strip()

            if (
                value.startswith("[")
                and value.endswith("]")
            ):
                value = (
                    value.removeprefix("[")
                    .removesuffix("]")
                )

            elif (
                value.startswith("(")
                and value.endswith(")")
            ):
                value = (
                    value.removeprefix("(")
                    .removesuffix(")")
                )

            return list(
                json.loads(
                    "[" + value + "]"
                )
            )

        type_factories = {
            "number": float,
            "integer": int,

            "boolean": (
                lambda value:
                value.lower()
                in {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "on",
                }
            ),

            "array": convert_array,
            "object": json.loads,
        }

        self.conversions: dict[
            str,
            dict[str, Any],
        ] = {
            function.name: {}
            for function in defs
        }

        for function in defs:

            for (
                parameter,
                definition,
            ) in function.parameters.items():

                self.conversions[
                    function.name
                ][parameter] = (
                    type_factories.get(
                        definition["type"],
                        str,
                    )
                )

    # ==================================================================
    # Token handling
    # ==================================================================

    def _add_token(
        self,
        output: list[int],
        state: Expect,
        token_id: int,
    ) -> None:
        """Add one selected token."""

        output.append(
            token_id
        )

        state.tokens.append(
            token_id
        )

        # --------------------------------------------------------------
        # ExpectValue has its own value decoder.
        # --------------------------------------------------------------

        if isinstance(
            state,
            ExpectValue,
        ):
            state.add_token(
                token_id
            )

        print(
            self.model.decode(
                [token_id]
            ),
            end="",
            flush=True,
        )

    # ==================================================================
    # Tokenization
    # ==================================================================

    def get_tokens(
        self,
        text: str,
    ) -> list[int]:
        """Encode text into token IDs."""

        return self.model.encode(
            text
        )[0].tolist()

    # ==================================================================
    # Inspection
    # ==================================================================

    def inspect(
        self,
        text: str,
    ) -> None:
        """Display the tokens making up text."""

        print()

        print(
            "Token    String"
        )

        print(
            "---------------------------"
        )

        for token_id in self.get_tokens(
            text
        ):

            print(
                f"{token_id:>8}  "
                f"{self.vocab[token_id]!r}"
            )

        print(
            "---------------------------"
        )

    # ==================================================================
    # Generation
    # ==================================================================
    def process_prompt(
    self,
    prompt: str,
    timeout: float = 30.0,
    examine: bool = False,
    ) -> str:
        """Generate a constrained function call."""

        print(
            f"\nPrompt\n{prompt}"
        )

        time_start = time.perf_counter()

        # --------------------------------------------------------------
        # Prompt context.
        # --------------------------------------------------------------

        context = (
            self.context
            + self.get_tokens(
                f'\nPrompt: "{prompt}"\n'
                f'JSON output: '
            )
        )

        # --------------------------------------------------------------
        # Fixed beginning of the JSON output.
        # --------------------------------------------------------------

        output = self.get_tokens(
            f'{{"prompt":"{prompt}",'
            f'"name":"'
        )

        # --------------------------------------------------------------
        # Start function generation.
        # --------------------------------------------------------------

        state: Expect | None = (
            ExpectFunction(
                self.state_context
            )
        )

        print(
            "\nResponse\n"
            f"{self.model.decode(output)}",
            end="",
            flush=True,
        )

        # ==============================================================
        # Generation loop
        # ==============================================================

        while (
            time.perf_counter()
            - time_start
            < timeout
        ):

            if state is None:
                break

            # ==========================================================
            # PARAMETER VALUE
            # ==========================================================

            if isinstance(
                state,
                ExpectValue,
            ):

                value_allowed = (
                    state.get_allowed()
                )

                if not value_allowed:

                    raise RuntimeError(
                        "No allowed tokens for "
                        "parameter value. "
                        f"Current value: "
                        f"{state.get_value()!r}"
                    )

                # ------------------------------------------------------
                # Structural termination is only available once the
                # value itself is syntactically complete.
                # ------------------------------------------------------

                termination_allowed = (
                    state.get_termination_tokens()
                )

                allowed = (
                    value_allowed
                    | termination_allowed
                )

                print(
                    "\n"
                    "[VALUE] "
                    f"type="
                    f"{state.parameter_type!r} "
                    f"value="
                    f"{state.get_value()!r} "
                    f"value_allowed="
                    f"{len(value_allowed)} "
                    f"termination_allowed="
                    f"{len(termination_allowed)}"
                )

                # ------------------------------------------------------
                # Ask model for logits.
                # ------------------------------------------------------

                logits = (
                    self.model
                    .get_logits_from_input_ids(
                        context + output
                    )
                )

                # ------------------------------------------------------
                # Keep only:
                #
                #   1. valid value continuation tokens
                #   2. structural termination token
                # ------------------------------------------------------

                for token_id in range(
                    len(logits)
                ):

                    if token_id not in allowed:

                        logits[token_id] = (
                            float("-inf")
                        )

                # ------------------------------------------------------
                # Select highest probability allowed token.
                # ------------------------------------------------------

                token_id = logits.index(
                    max(logits)
                )

                # ------------------------------------------------------
                # If the model selects the structural token,
                # the value is finished.
                #
                # IMPORTANT:
                #
                # We do NOT send this token into the
                # ParameterValueRule.
                # It belongs to the structural state.
                # ------------------------------------------------------

                if (
                    token_id
                    in termination_allowed
                ):

                    print(
                        "\n"
                        "[VALUE COMPLETE] "
                        f"{state.get_value()!r}"
                    )

                    state = (
                        state.next_state()
                    )

                    # --------------------------------------------------
                    # The termination token itself must now be
                    # processed by the new structural state.
                    #
                    # Add it to the structural output/state.
                    # --------------------------------------------------
    #                 self._add_token(
    #     output,
    #     state,
    #     token_id,
    # )
                    output.append(token_id)
                    state.tokens.append(token_id)
                    print(self.model.decode([token_id]), end="", flush=True, )

                    continue

                # ------------------------------------------------------
                # Otherwise this is a normal value token.
                # ------------------------------------------------------

                check = state.check_token(
                    token_id
                )

                if not check.accepted:

                    raise RuntimeError(
                        "Selected token was not "
                        "accepted by value rule."
                    )

                self._add_token(
                    output,
                    state,
                    token_id,
                )

                continue

            # ==========================================================
            # STRUCTURAL STATES
            # ==========================================================

            allowed = (
                state.get_allowed()
            )

            # ----------------------------------------------------------
            # Current structural sequence is complete.
            # ----------------------------------------------------------

            if not allowed:

                state = (
                    state.next_state()
                )

                continue

            # ----------------------------------------------------------
            # Only one structural token is possible.
            # ----------------------------------------------------------

            if len(allowed) == 1:

                token_id = next(
                    iter(allowed)
                )

                self._add_token(
                    output,
                    state,
                    token_id,
                )

                continue

            # ----------------------------------------------------------
            # More than one structural token is possible.
            # ----------------------------------------------------------

            logits = (
                self.model
                .get_logits_from_input_ids(
                    context + output
                )
            )

            for token_id in range(
                len(logits)
            ):

                if token_id not in allowed:

                    logits[token_id] = (
                        float("-inf")
                    )

            token_id = logits.index(
                max(logits)
            )

            self._add_token(
                output,
                state,
                token_id,
            )

        else:

            raise RuntimeError(
                f"Time limit ({timeout}) reached."
            )

        # ==============================================================
        # Finished
        # ==============================================================

        result = self.model.decode(
            output
        )

        elapsed = (
            time.perf_counter()
            - time_start
        )

        print(
            "\n\nResponse finished in "
            f"{round(elapsed)} seconds."
        )

        if examine:
            self.inspect(
                result
            )

        return result

    # ==================================================================
    # Dump
    # ==================================================================

    def dump(
        self,
        text: str,
        path: Path,
    ) -> None:
        """Dump the generated function call."""

        new_call = FunctionCall(
            **json.loads(text)
        )

        converted_params = {}

        for (
            key,
            value,
        ) in new_call.parameters.items():

            try:

                converted_params[key] = (
                    self.conversions[
                        new_call.name
                    ][key](value)
                )

            except ValueError:

                converted_params[key] = value

        function_calls = [
            {
                "prompt": new_call.prompt,
                "name": new_call.name,
                "parameters": converted_params,
            }
        ]

        path.parent.mkdir(
            exist_ok=True
        )

        try:

            function_calls = (
                json.loads(
                    path.read_text()
                )
                + function_calls
            )

        except (
            json.JSONDecodeError,
            OSError,
            TypeError,
        ):
            pass

        with path.open("w") as file:

            json.dump(
                function_calls,
                file,
                indent="\t",
            )