import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


INTEGER_START = re.compile(
    r"-?(0|[1-9][0-9]*)?$"
)

INTEGER_DONE = re.compile(
    r"-?(0|[1-9][0-9]*)$"
)

NUMBER_START = re.compile(
    r"-?(0|[1-9][0-9]*)(\.[0-9]*)?$"
)

NUMBER_DONE = re.compile(
    r"-?(0|[1-9][0-9]*)(\.[0-9]+)?$"
)


class ValueCheck(BaseModel):
    """Result of checking a possible next token."""

    accepted: bool
    value: str
    tail: str = ""
    finished: bool = False


class ParameterValueRule(ABC):
    """Rules used to constrain generation of one parameter value."""

    def __init__(
        self,
        model: Any,
        vocabulary: list[str],
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.token_ids: list[int] = []

    def add_token(
        self,
        token_id: int,
    ) -> None:
        """Record a token that the model actually selected."""

        self.token_ids.append(
            token_id
        )

    def get_value(self) -> str:
        """Return the text generated for this value so far."""

        return self.model.decode(
            self.token_ids
        )

    def try_token(
        self,
        token_id: int,
    ) -> ValueCheck:
        """Check what would happen if token_id were selected."""

        ids = (
            self.token_ids
            + [token_id]
        )

        text = self.model.decode(
            ids
        )

        return self.inspect(
            text
        )

    @abstractmethod
    def inspect(
        self,
        text: str,
    ) -> ValueCheck:
        """Describe whether text can still represent this value."""

        raise NotImplementedError

    def get_allowed_tokens(self) -> set[int]:
        """
        Return vocabulary tokens that keep the value valid.

        A token is allowed when adding it produces an
        accepted ValueCheck.
        """

        result: set[int] = set()

        for token_id in range(
            len(self.vocabulary)
        ):
            check = self.try_token(
                token_id
            )

            if check.accepted:
                result.add(
                    token_id
                )

        return result

    def get_finished_tokens(self) -> set[int]:
        """
        Return tokens which produce a complete value.
        """

        result: set[int] = set()

        for token_id in range(
            len(self.vocabulary)
        ):
            check = self.try_token(
                token_id
            )

            if (
                check.accepted
                and check.finished
            ):
                result.add(
                    token_id
                )

        return result

    def get_token_check(
        self,
        token_id: int,
    ) -> ValueCheck:
        """
        Inspect a possible next token without
        actually adding it.
        """

        return self.try_token(
            token_id
        )

    def can_finish(self) -> bool:
        """
        Return whether the current value is
        syntactically complete.

        This does NOT mean generation must stop.
        """

        result = self.inspect(
            self.get_value()
        )

        return (
            result.accepted
            and result.finished
        )


class IntegerParameter(
    ParameterValueRule
):
    """Rules for JSON integer values."""

    def inspect(
        self,
        text: str,
    ) -> ValueCheck:

        if text == "":
            return ValueCheck(
                accepted=True,
                value=text,
            )

        if INTEGER_DONE.fullmatch(
            text
        ):
            return ValueCheck(
                accepted=True,
                value=text,
                finished=True,
            )

        for delimiter in (
            ",",
            "}",
        ):
            if text.endswith(
                delimiter
            ):
                value = text[:-1]

                if INTEGER_DONE.fullmatch(
                    value
                ):
                    return ValueCheck(
                        accepted=True,
                        value=value,
                        tail=delimiter,
                        finished=True,
                    )

        if INTEGER_START.fullmatch(
            text
        ):
            return ValueCheck(
                accepted=True,
                value=text,
            )

        return ValueCheck(
            accepted=False,
            value=text,
        )


class NumberParameter(
    ParameterValueRule
):
    """Rules for JSON number values."""

    def inspect(
        self,
        text: str,
    ) -> ValueCheck:

        if text == "":
            return ValueCheck(
                accepted=True,
                value=text,
            )

        if NUMBER_DONE.fullmatch(
            text
        ):
            return ValueCheck(
                accepted=True,
                value=text,
                finished=True,
            )

        if (
            text == "-"
            or NUMBER_START.fullmatch(
                text
            )
        ):
            return ValueCheck(
                accepted=True,
                value=text,
            )

        for delimiter in (
            ",",
            "}",
        ):
            if text.endswith(
                delimiter
            ):
                value = text[:-1]

                if NUMBER_DONE.fullmatch(
                    value
                ):
                    return ValueCheck(
                        accepted=True,
                        value=value,
                        tail=delimiter,
                        finished=True,
                    )

        return ValueCheck(
            accepted=False,
            value=text,
        )


class BooleanParameter(
    ParameterValueRule
):
    """Rules for JSON true/false values."""

    WORDS = (
        "true",
        "false",
    )

    def inspect(
        self,
        text: str,
    ) -> ValueCheck:

        if text == "":
            return ValueCheck(
                accepted=True,
                value=text,
            )

        for word in self.WORDS:

            if text == word:
                return ValueCheck(
                    accepted=True,
                    value=text,
                    finished=True,
                )

            if word.startswith(
                text
            ):
                return ValueCheck(
                    accepted=True,
                    value=text,
                )

            for delimiter in (
                ",",
                "}",
            ):
                if text == (
                    word + delimiter
                ):
                    return ValueCheck(
                        accepted=True,
                        value=word,
                        tail=delimiter,
                        finished=True,
                    )

        return ValueCheck(
            accepted=False,
            value=text,
        )


class StringParameter(
    ParameterValueRule
):
    """
    Rules for JSON string values.

    The surrounding JSON quotes are NOT part of
    the value generated by this rule.

    The structural state machine is responsible
    for generating the JSON syntax around the value.

    Example:

        ExpectKVSep -> '":"'
        StringParameter -> 'Shrek'
        ExpectSep -> '","'
    """

    def inspect(
        self,
        text: str,
    ) -> ValueCheck:

        # ----------------------------------------------------------
        # Empty string.
        #
        # At this point no value characters have been generated.
        # The empty string is valid, but it is not finished yet.
        # ----------------------------------------------------------

        if text == "":
            return ValueCheck(
                accepted=True,
                value=text,
                finished=False,
            )

        # ----------------------------------------------------------
        # A raw quote is NOT part of the string value.
        #
        # If the tokenizer produces a quote token here, reject it.
        # The structural state owns those quotes.
        # ----------------------------------------------------------

        if '"' in text:
            return ValueCheck(
                accepted=False,
                value=text,
            )

        # ----------------------------------------------------------
        # JSON strings cannot contain unescaped control characters.
        # ----------------------------------------------------------

        for char in text:

            if ord(char) < 0x20:
                return ValueCheck(
                    accepted=False,
                    value=text,
                )

        # ----------------------------------------------------------
        # Everything else is a valid string prefix.
        #
        # We intentionally consider every non-empty prefix valid.
        #
        # The string is complete when the current generation step
        # chooses the structural separator/end token.
        # ----------------------------------------------------------

        return ValueCheck(
            accepted=True,
            value=text,
            finished=True,
        )


class JsonContainerParameter(
    ParameterValueRule
):
    """
    Conservative JSON-prefix checking
    for arrays and objects.
    """

    opening: str
    closing: str
    python_type: type

    def inspect(
        self,
        text: str,
    ) -> ValueCheck:

        if text == "":
            return ValueCheck(
                accepted=True,
                value=text,
            )

        if not text.startswith(
            self.opening
        ):
            return ValueCheck(
                accepted=False,
                value=text,
            )

        # ----------------------------------------------------------
        # Complete JSON container.
        # ----------------------------------------------------------

        try:

            value = json.loads(
                text
            )

            if isinstance(
                value,
                self.python_type,
            ):
                return ValueCheck(
                    accepted=True,
                    value=text,
                    finished=True,
                )

        except json.JSONDecodeError:
            pass

        # ----------------------------------------------------------
        # Container followed by structural delimiter.
        # ----------------------------------------------------------

        for delimiter in (
            ",",
            "}",
        ):

            if text.endswith(
                delimiter
            ):

                value_text = text[:-1]

                try:

                    value = json.loads(
                        value_text
                    )

                    if isinstance(
                        value,
                        self.python_type,
                    ):
                        return ValueCheck(
                            accepted=True,
                            value=value_text,
                            tail=delimiter,
                            finished=True,
                        )

                except json.JSONDecodeError:
                    pass

        return self._possible_prefix(
            text
        )

    def _possible_prefix(
        self,
        text: str,
    ) -> ValueCheck:

        depth = 0
        in_string = False
        escaped = False

        for char in text:

            if in_string:

                if escaped:
                    escaped = False

                elif char == "\\":
                    escaped = True

                elif char == '"':
                    in_string = False

                elif ord(char) < 0x20:
                    return ValueCheck(
                        accepted=False,
                        value=text,
                    )

                continue

            if char == '"':
                in_string = True

            elif char == self.opening:
                depth += 1

            elif char == self.closing:
                depth -= 1

                if depth < 0:
                    return ValueCheck(
                        accepted=False,
                        value=text,
                    )

        if (
            depth >= 1
            or in_string
        ):
            return ValueCheck(
                accepted=True,
                value=text,
            )

        return ValueCheck(
            accepted=False,
            value=text,
        )


class ArrayParameter(
    JsonContainerParameter
):
    opening = "["
    closing = "]"
    python_type = list


class ObjectParameter(
    JsonContainerParameter
):
    opening = "{"
    closing = "}"
    python_type = dict


class ParameterValueFactory:
    """Build the value rules required by a parameter schema."""

    TYPES = {
        "integer": IntegerParameter,
        "number": NumberParameter,
        "boolean": BooleanParameter,
        "string": StringParameter,
        "array": ArrayParameter,
        "object": ObjectParameter,
    }

    @classmethod
    def build(
        cls,
        parameter_type: str,
        model: Any,
        vocabulary: list[str],
    ) -> ParameterValueRule:

        try:

            rule_type = cls.TYPES[
                parameter_type
            ]

        except KeyError as exc:

            raise ValueError(
                "Unsupported parameter type: "
                f"{parameter_type}"
            ) from exc

        return rule_type(
            model,
            vocabulary,
        )