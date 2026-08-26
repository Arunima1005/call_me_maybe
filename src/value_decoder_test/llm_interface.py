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

    MAX_VALUE_TOKENS = 32

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
        # Function definitions.
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
        # Shared state context.
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
        # Conversion functions.
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
        """Display tokens making up text."""

        print()
        print("Token    String")
        print("---------------------------")

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

    def _best_token(
        self,
        logits: list[float],
        allowed: set[int],
    ) -> int:
        """Return the highest-logit token from allowed tokens."""

        if not allowed:
            raise RuntimeError(
                "No allowed tokens."
            )

        return max(
            allowed,
            key=lambda token_id:
            logits[token_id],
        )


    def process_prompt(
    self,
    prompt: str,
    timeout: float = 30.0,
    examine: bool = False,
) -> str:
        """Generate one constrained function call."""

        start_time = time.monotonic()

        # --------------------------------------------------------------
        # Build model input.
        # --------------------------------------------------------------

        generated_ids = self.context.copy()

        prompt_text = str(
            {"prompt": prompt}
        )

        generated_ids.extend(
            self.get_tokens(
                prompt_text
            )
        )

        # --------------------------------------------------------------
        # Start state machine.
        # --------------------------------------------------------------

        state: Expect = ExpectFunction(
            self.state_context
        )

        # Only the constrained function-call output.
        output_ids: list[int] = []

        value_token_count = 0

        while True:

            # ----------------------------------------------------------
            # Timeout.
            # ----------------------------------------------------------

            if (
                time.monotonic()
                - start_time
                > timeout
            ):
                raise RuntimeError(
                    f"Time limit ({timeout}s) reached."
                )

            # ----------------------------------------------------------
            # Ask state what is allowed.
            # ----------------------------------------------------------

            allowed = state.get_allowed()

            # ----------------------------------------------------------
            # No tokens means this state is complete.
            # Move to next state.
            # ----------------------------------------------------------

            if not allowed:

                next_state = state.next_state()

                if next_state is None:
                    break

                state = next_state

                if isinstance(
                    state,
                    ExpectValue,
                ):
                    value_token_count = 0

                continue

            # ----------------------------------------------------------
            # Get logits for next token.
            # ----------------------------------------------------------

            logits = (
                self.model.get_logits_from_input_ids(
                    generated_ids
                )
            )

            # ==========================================================
            # VALUE STATE
            # ==========================================================

            if isinstance(
                state,
                ExpectValue,
            ):

                termination_tokens = (
                    state.get_termination_tokens()
                )

                value_tokens = allowed

                force_termination = (
                    value_token_count
                    >= self.MAX_VALUE_TOKENS
                )

                # ------------------------------------------------------
                # Decide between:
                #
                #   1. continuing the value
                #   2. terminating the value
                # ------------------------------------------------------

                if (
                    state.can_finish()
                    and termination_tokens
                ):

                    best_termination_token = (
                        self._best_token(
                            logits,
                            termination_tokens,
                        )
                    )

                    termination_logit = (
                        logits[
                            best_termination_token
                        ]
                    )

                    best_value_token = None

                    best_value_logit = (
                        float("-inf")
                    )

                    if value_tokens:

                        best_value_token = (
                            self._best_token(
                                logits,
                                value_tokens,
                            )
                        )

                        best_value_logit = (
                            logits[
                                best_value_token
                            ]
                        )

                    # --------------------------------------------------
                    # Terminate if:
                    #
                    # - value is too long
                    # - no value token exists
                    # - termination is more likely
                    # --------------------------------------------------

                    if (
                        force_termination
                        or best_value_token is None
                        or termination_logit >= best_value_logit
                    ):

                        token_id = best_termination_token

                        generated_ids.append(token_id)
                        output_ids.append(token_id)

                        print(
                            self.model.decode([token_id]),
                            end="",
                            flush=True,
                        )

                        # The termination token has already been generated.
                        # It is also the first token of the next structural state.
                        next_state = state.next_state()

                        if next_state is None:
                            break

                        next_state.tokens.append(token_id)

                        state = next_state

                        value_token_count = 0

                        if examine:
                            print(
                                "\n[VALUE COMPLETE]"
                                f" {state.context.current_parameter}"
                            )

                        continue

                    token_id = (
                        best_value_token
                    )

                else:

                    # --------------------------------------------------
                    # Value is not complete.
                    # Only value tokens are allowed.
                    # --------------------------------------------------

                    if not value_tokens:
                        raise RuntimeError(
                            "No allowed tokens for "
                            "parameter value."
                        )

                    token_id = (
                        self._best_token(
                            logits,
                            value_tokens,
                        )
                    )

                # ------------------------------------------------------
                # Add value token.
                # ------------------------------------------------------

                generated_ids.append(
                    token_id
                )

                output_ids.append(
                    token_id
                )

                state.add_token(
                    token_id
                )

                value_token_count += 1

                print(
                    self.model.decode(
                        [token_id]
                    ),
                    end="",
                    flush=True,
                )

                if examine:

                    print(
                        "\n[VALUE]"
                        f" type={state.parameter.type!r}"
                        f" value={state.get_value()!r}"
                        f" value_allowed="
                        f"{len(state.get_allowed())}"
                        f" termination_allowed="
                        f"{len(state.get_termination_tokens())}"
                    )

                continue

            # ==========================================================
            # STRUCTURAL STATE
            # ==========================================================

            token_id = (
                self._best_token(
                    logits,
                    allowed,
                )
            )

            # ----------------------------------------------------------
            # Add structural token to model input/output.
            # ----------------------------------------------------------

            generated_ids.append(
                token_id
            )

            output_ids.append(
                token_id
            )

            # ----------------------------------------------------------
            # Add token to current structural state.
            # ----------------------------------------------------------

            state.tokens.append(
                token_id
            )

            print(
                self.model.decode(
                    [token_id]
                ),
                end="",
                flush=True,
            )

            if examine:
                print(
                    "\n[STATE]"
                    f" {type(state).__name__}"
                    f" tokens={state.tokens!r}"
                )

        # --------------------------------------------------------------
        # Build complete JSON.
        #
        # The constrained decoder generates:
        #
        #     fn_add_numbers","parameters":{"a":"2","b":"3"}}
        #
        # The deterministic prefix is added here.
        # --------------------------------------------------------------

        generated_text = self.model.decode(
            output_ids
        )

        prefix = (
            '{"prompt":'
            + json.dumps(prompt)
            + ',"name":"'
        )

        result = (
            prefix
            + generated_text
        )

        if examine:
            print(
                "\n\n[FINAL GENERATED TEXT]"
            )
            print(
                repr(result)
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
        """Dump generated function call."""

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

            except (
                ValueError,
                TypeError,
            ):
                converted_params[key] = value

        function_calls = [
            {
                "prompt": new_call.prompt,
                "name": new_call.name,
                "parameters": converted_params,
            }
        ]

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
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