
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

INTEGER_PREFIX = re.compile(r"(?:-?|0|[1-9][0-9]*)$")
INTEGER_COMPLETE = re.compile(r"-?(0|[1-9][0-9]*)$")

NUMBER_PREFIX = re.compile(r"(?:-?|0|[1-9][0-9]*)(?:\.[0-9]*)?$")
NUMBER_COMPLETE = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?$")


@dataclass(frozen=True)
class Candidate:
    valid: bool
    value: str
    remainder: str = ""
    complete: bool = False


class ValueDecoder(ABC):
    """Validate one JSON parameter value against vocabulary tokens."""

    def __init__(self, model: Any, vocabulary: list[str]) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.generated_ids: list[int] = []

    def add_token(self, token_id: int) -> None:
        self.generated_ids.append(token_id)

    def get_value(self) -> str:
        return self.model.decode(self.generated_ids)

    def candidate_text(self, token_id: int) -> str:
        return self.model.decode(self.generated_ids + [token_id])

    @abstractmethod
    def analyse(self, text: str) -> Candidate:
        """Analyse a candidate token sequence."""
        raise NotImplementedError

    def get_allowed_tokens(self) -> set[int]:
        allowed: set[int] = set()
        for token_id in range(len(self.vocabulary)):
            result = self.analyse(self.candidate_text(token_id))
            if result.valid:
                allowed.add(token_id)
        return allowed

    def can_finish(self) -> bool:
        result = self.analyse(self.get_value())
        return result.valid and result.complete


class IntegerDecoder(ValueDecoder):
    """JSON integer value decoder."""

    def analyse(self, text: str) -> Candidate:
        if text == "":
            return Candidate(True, "", complete=False)

        match = INTEGER_COMPLETE.fullmatch(text)
        if match:
            return Candidate(True, text, complete=True)

        if INTEGER_PREFIX.fullmatch(text):
            return Candidate(True, text, complete=False)

        # A tokenizer may produce a value plus a structural delimiter
        # in one token, e.g. "125," or "125}".
        for delimiter in (",", "}"):
            if text.endswith(delimiter):
                value = text[:-1]
                if INTEGER_COMPLETE.fullmatch(value):
                    return Candidate(True, value, delimiter, True)

        return Candidate(False, text)


class NumberDecoder(ValueDecoder):
    """JSON number value decoder."""

    def analyse(self, text: str) -> Candidate:
        if text == "":
            return Candidate(True, "", complete=False)

        match = NUMBER_COMPLETE.fullmatch(text)
        if match:
            return Candidate(True, text, complete=True)

        if NUMBER_PREFIX.fullmatch(text):
            return Candidate(True, text, complete=False)

        for delimiter in (",", "}"):
            if text.endswith(delimiter):
                value = text[:-1]
                if NUMBER_COMPLETE.fullmatch(value):
                    return Candidate(True, value, delimiter, True)

        return Candidate(False, text)


class BooleanDecoder(ValueDecoder):
    """JSON boolean value decoder."""

    TRUE = "true"
    FALSE = "false"

    def analyse(self, text: str) -> Candidate:
        if text == "":
            return Candidate(True, "", complete=False)

        for value in (self.TRUE, self.FALSE):
            if text == value:
                return Candidate(True, value, complete=True)
            if value.startswith(text):
                return Candidate(True, text, complete=False)
            if text == value + ",":
                return Candidate(True, value, ",", True)
            if text == value + "}":
                return Candidate(True, value, "}", True)

        return Candidate(False, text)


class StringDecoder(ValueDecoder):
    """JSON string decoder. Generated text includes JSON quotes."""

    def analyse(self, text: str) -> Candidate:
        if text == "":
            return Candidate(True, "", complete=False)

        if not text.startswith('"'):
            return Candidate(False, text)

        # A complete JSON string, optionally followed by a delimiter.
        try:
            value = json.loads(text)
            if isinstance(value, str):
                return Candidate(True, text, complete=True)
        except json.JSONDecodeError:
            pass

        for delimiter in (",", "}"):
            if text.endswith(delimiter):
                value_text = text[:-1]
                try:
                    value = json.loads(value_text)
                    if isinstance(value, str):
                        return Candidate(
                            True,
                            value_text,
                            delimiter,
                            True,
                        )
                except json.JSONDecodeError:
                    pass

        # Still inside a string. A JSON string prefix must have an
        # opening quote and no unescaped closing quote.
        escaped = False
        for char in text[1:]:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return Candidate(False, text)

            if ord(char) < 0x20:
                return Candidate(False, text)

        # A trailing backslash is a valid prefix of an escape sequence.
        return Candidate(True, text, complete=False)


class JsonValueDecoder(ValueDecoder):
    """
    Conservative decoder for JSON arrays/objects.

    Full element-level type constraints can be added later. This class
    first guarantees that the generated text remains a possible JSON
    container prefix.
    """

    opening: str
    closing: str
    expected_type: type

    def analyse(self, text: str) -> Candidate:
        if text == "":
            return Candidate(True, "", complete=False)

        if not text.startswith(self.opening):
            return Candidate(False, text)

        try:
            value = json.loads(text)
            if isinstance(value, self.expected_type):
                return Candidate(True, text, complete=True)
        except json.JSONDecodeError:
            pass

        # Handle value + function-call delimiter in one token.
        for delimiter in (",", "}"):
            if text.endswith(delimiter):
                value_text = text[:-1]
                try:
                    value = json.loads(value_text)
                    if isinstance(value, self.expected_type):
                        return Candidate(True, value_text, delimiter, True)
                except json.JSONDecodeError:
                    pass

        return self._balanced_prefix(text)

    def _balanced_prefix(self, text: str) -> Candidate:
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
                    return Candidate(False, text)
                continue

            if char == '"':
                in_string = True
            elif char == self.opening:
                depth += 1
            elif char == self.closing:
                depth -= 1
                if depth < 0:
                    return Candidate(False, text)

        if depth >= 1 or in_string:
            return Candidate(True, text, complete=False)

        return Candidate(False, text)


class ArrayDecoder(JsonValueDecoder):
    opening = "["
    closing = "]"
    expected_type = list


class ObjectDecoder(JsonValueDecoder):
    opening = "{"
    closing = "}"
    expected_type = dict


class ValueDecoderFactory:
    DECODERS = {
        "integer": IntegerDecoder,
        "number": NumberDecoder,
        "boolean": BooleanDecoder,
        "string": StringDecoder,
        "array": ArrayDecoder,
        "object": ObjectDecoder,
    }

    @classmethod
    def create(
        cls,
        parameter_type: str,
        model: Any,
        vocabulary: list[str],
    ) -> ValueDecoder:
        try:
            decoder_class = cls.DECODERS[parameter_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported parameter type: {parameter_type}"
            ) from exc

        return decoder_class(model, vocabulary)
