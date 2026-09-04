import json
import time
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model

from .data_models import (
    FunctionDefinition,
    FunctionCall,
    ParameterDefinition,
    StateContext,
)
from .decoder import ValueDecoder
from .expect import (
    Expect,
    ExpectFunction,
    ExpectValue,
)
from .formatting import X, H, U, C, D


class LLMInterface:
    """
    Interface for constrained function-call generation.
    """

    def __init__(
        self,
        model_name: str,
        defs: list[FunctionDefinition],
    ) -> None:

        print(
            f"{H + U + C}Current model{X}: "
            f"{H + model_name + X}\n"
        )

        self.model = Small_LLM_Model(model_name)

        self.vocab: list[str] = list(
            json.loads(
                Path(
                    self.model.get_path_to_vocab_file()
                ).read_text()
            ).keys()
        )

        self.context = self.get_tokens(
            f"""Available functions:
{'\n'.join(
    [
        ': '.join(
            [x.name, x.description]
        )
        for x in defs
    ]
)}"""
        )

        if defs:
            print(
                f"{H + U + C}Context{X}\n"
                f"{self.model.decode(self.context)}"
            )

        # --------------------------------------------------------------
        # Encode structural pieces
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
                +
                self.get_tokens(
                    '","'
                )
            )

            parameters = [
                ParameterDefinition(
                    name=name,
                    type=content["type"],
                )
                for name, content
                in function.parameters.items()
            ]

            functions[function_tokens] = parameters

        self.state_context = StateContext(
            model=self.model,
            vocabulary=self.vocab,
            encode=self.get_tokens,
            functions=functions,

            param_def=tuple(
                self.get_tokens(
                    'parameters":{'
                )
            ),

            # IMPORTANT:
            #
            # This is ":" and NOT '":"'.
            #
            # StringDecoder will generate the quote
            # because quotes are part of the string value.
            kvsep=tuple(
                self.get_tokens(':')
            ),

            sep=tuple(
                self.get_tokens(',')
            ),

            end=tuple(
                self.get_tokens(
                    '}}\n'
                )
            ),
        )

    # ==================================================================
    # Token helpers
    # ==================================================================

    def get_tokens(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        """

        return self.model.encode(text)[0].tolist()

    def inspect(self, text: str) -> None:
        """
        Print token information.
        """

        print(
            f"""{D + "─" * 9}┬{"─" * 30 + X}
{H}Token{X}    {D}│{X} {H}String{X}
{D + "─" * 9}┼{"─" * 30 + X}"""
        )

        for token in self.get_tokens(text):

            print(
                f"{token:>8} "
                f"{D}│{X} "
                f"{self.vocab[token]}"
            )

        print(
            f"{D + "─" * 9}┴{"─" * 30 + X}"
        )

    # ==================================================================
    # Choose token
    # ==================================================================

    def choose_token(
        self,
        context: list[int],
        output: list[int],
        allowed: set[int],
    ) -> int:
        """
        Ask the model for logits and return the highest-scoring
        allowed token.
        """

        logits = self.model.get_logits_from_input_ids(
            context + output
        )

        best_token = None
        best_score = float("-inf")

        for token_id in allowed:

            score = logits[token_id]

            if score > best_score:
                best_score = score
                best_token = token_id

        if best_token is None:
            raise RuntimeError(
                "No allowed token has a valid logit."
            )

        return best_token

    # ==================================================================
    # Process prompt
    # ==================================================================

    def process_prompt(
        self,
        prompt: str,
        timeout: float = 30.0,
        examine: bool = False,
    ) -> str:

        print(
            f"{H + U + C}\nPrompt{X}\n{prompt}"
        )

        time_start = time.perf_counter()

        # --------------------------------------------------------------
        # Prompt context
        # --------------------------------------------------------------

        context = (
            self.context
            +
            self.get_tokens(
                f'\nPrompt: "{prompt}"\nJSON output: '
            )
        )

        # --------------------------------------------------------------
        # Beginning of JSON output
        # --------------------------------------------------------------

        output = self.get_tokens(
            f'{{"prompt":"{prompt}","name":"'
        )

        # --------------------------------------------------------------
        # Initial state
        # --------------------------------------------------------------

        state: Expect | None = ExpectFunction(
            self.state_context
        )

        print(
            f"{H + U + C}Response{X}\n"
            f"{D + self.model.decode(output) + X}",
            end="",
        )

        # ==============================================================
        # Generation loop
        # ==============================================================

        while time.perf_counter() - time_start < timeout:

            if state is None:
                break

            # ----------------------------------------------------------
            # VALUE STATE
            # ----------------------------------------------------------

            if isinstance(state, ExpectValue):

                allowed = state.get_allowed()

                if not allowed:
                    raise RuntimeError(
                        "No valid tokens available for "
                        "parameter value."
                    )

                token_id = self.choose_token(
                    context,
                    output,
                    allowed,
                )

                output.append(token_id)

                state.add_token(token_id)

                token_text = self.vocab[token_id]

                print(
                    token_text,
                    end="",
                    flush=True,
                )

                # ------------------------------------------------------
                # Important:
                #
                # We only leave the value when it is complete.
                # ------------------------------------------------------

                if state.can_finish():

                    next_state = state.next_state()

                    state = next_state

                continue

            # ----------------------------------------------------------
            # STRUCTURAL STATE
            # ----------------------------------------------------------

            allowed = state.get_allowed()

            if not allowed:

                next_state = state.next_state()

                state = next_state

                continue

            token_id = self.choose_token(
                context,
                output,
                allowed,
            )

            output.append(token_id)

            if hasattr(state, "tokens"):
                state.tokens.append(token_id)

            print(
                self.vocab[token_id],
                end="",
                flush=True,
            )

            # ----------------------------------------------------------
            # Check whether structural sequence is complete.
            # ----------------------------------------------------------

            expected = getattr(
                state,
                "expected",
                None,
            )

            if expected is not None:

                if tuple(state.tokens) == expected:

                    state = state.next_state()

            elif isinstance(
                state,
                ExpectFunction,
            ):

                # Function selection is complete when the generated
                # token sequence matches one of the available functions.

                if tuple(state.tokens) in (
                    state.context.functions
                ):
                    state = state.next_state()

        else:

            raise RuntimeError(
                f"Time limit ({timeout}) reached."
            )

        print(
            f"\n{H}Response finished in "
            f"{round(time.perf_counter() - time_start)} "
            f"seconds.{X}"
        )

        result = self.model.decode(output)

        if examine:
            self.inspect(result)

        return result

    # ==================================================================
    # Dump
    # ==================================================================

    def dump(
        self,
        text: str,
        path: Path,
    ) -> None:

        new_call = FunctionCall(
            **json.loads(text)
        )

        # --------------------------------------------------------------
        # Type conversions
        # --------------------------------------------------------------

        conversions: dict[
            str,
            dict[str, Any],
        ] = {}

        # Build conversion information from the original
        # function definitions if needed.
        #
        # This section can remain similar to your existing
        # implementation.
        #
        # For now JSON has already been validated by FunctionCall.

        function_calls = [{
            "prompt": new_call.prompt,
            "name": new_call.name,
            "parameters": new_call.parameters,
        }]

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            existing = json.loads(
                path.read_text()
            )

            if isinstance(existing, list):
                function_calls = (
                    existing + function_calls
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