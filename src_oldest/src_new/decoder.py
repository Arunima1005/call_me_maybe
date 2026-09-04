import json
import re
from abc import ABC, abstractmethod
from typing import Any


# ============================================================================
# Regular expressions
# ============================================================================

INTEGER_PREFIX = re.compile(
    r"-?(0|[1-9][0-9]*)$"
)

INTEGER_COMPLETE = re.compile(
    r"-?(0|[1-9][0-9]*)$"
)


NUMBER_PREFIX = re.compile(
    r"-?(0|[1-9][0-9]*)(\.[0-9]*)?$"
)

NUMBER_COMPLETE = re.compile(
    r"-?(0|[1-9][0-9]*)(\.[0-9]+)?$"
)


# ============================================================================
# Base ValueDecoder
# ============================================================================

class ValueDecoder(ABC):
    """
    Base class for parameter-value constrained decoding.

    The decoder answers:

        "Which vocabulary tokens can legally come next?"

    It does NOT choose the token.

    The LLM chooses the highest-scoring token from the allowed set.
    """

    def __init__(
        self,
        model: Any,
        vocabulary: list[str],
    ) -> None:

        self.model = model
        self.vocabulary = vocabulary

        # Tokens that have ACTUALLY been generated.
        self.generated_ids: list[int] = []

    # ------------------------------------------------------------------
    # Generated tokens
    # ------------------------------------------------------------------

    def add_token(self, token_id: int) -> None:
        """
        Add a token after the LLM has actually selected it.
        """

        self.generated_ids.append(token_id)

    def reset(self) -> None:
        """Reset the decoder for a new value."""

        self.generated_ids.clear()

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def get_value(self) -> str:
        """Decode the currently generated value."""

        return self.model.decode(self.generated_ids)

    def candidate_value(self, token_id: int) -> str:
        """
        Decode the value that would exist if token_id were selected.

        IMPORTANT:

        This does not modify generated_ids.
        """

        candidate_ids = self.generated_ids + [token_id]

        return self.model.decode(candidate_ids)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @abstractmethod
    def is_valid_prefix(self, value: str) -> bool:
        """
        Can this text still become a valid value?
        """

        raise NotImplementedError

    @abstractmethod
    def is_complete(self, value: str) -> bool:
        """
        Is this already a complete value?
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Allowed tokens
    # ------------------------------------------------------------------

    def get_allowed_tokens(self) -> set[int]:
        """
        Return vocabulary tokens that keep the value valid.
        """

        allowed: set[int] = set()

        for token_id in range(len(self.vocabulary)):

            candidate = self.candidate_value(token_id)

            if self.is_valid_prefix(candidate):
                allowed.add(token_id)

        return allowed

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def can_finish(self) -> bool:
        """Return True if the currently generated value is complete."""

        return self.is_complete(self.get_value())


# ============================================================================
# IntegerDecoder
# ============================================================================

class IntegerDecoder(ValueDecoder):
    """
    Decode JSON integer values.

    Examples:

        0
        1
        42
        -1
        -250
    """

    def is_valid_prefix(self, value: str) -> bool:

        if value == "":
            return True

        return INTEGER_PREFIX.fullmatch(value) is not None

    def is_complete(self, value: str) -> bool:

        if value == "":
            return False

        return INTEGER_COMPLETE.fullmatch(value) is not None


# ============================================================================
# NumberDecoder
# ============================================================================

class NumberDecoder(ValueDecoder):
    """
    Decode JSON number values.

    Examples:

        0
        10
        -10
        10.5
        -10.50
    """

    def is_valid_prefix(self, value: str) -> bool:

        if value == "":
            return True

        return NUMBER_PREFIX.fullmatch(value) is not None

    def is_complete(self, value: str) -> bool:

        if value == "":
            return False

        return NUMBER_COMPLETE.fullmatch(value) is not None


# ============================================================================
# BooleanDecoder
# ============================================================================

class BooleanDecoder(ValueDecoder):
    """
    Decode JSON booleans.

    Valid values:

        true
        false
    """

    TRUE = "true"
    FALSE = "false"

    def is_valid_prefix(self, value: str) -> bool:

        if value == "":
            return True

        return (
            self.TRUE.startswith(value)
            or self.FALSE.startswith(value)
        )

    def is_complete(self, value: str) -> bool:

        return value in {
            self.TRUE,
            self.FALSE,
        }


# ============================================================================
# StringDecoder
# ============================================================================

class StringDecoder(ValueDecoder):
    """
    Decode JSON strings.

    The generated value includes its quotes.

    Examples:

        "EUR"
        "hello"
        "payment reference"
    """

    def is_valid_prefix(self, value: str) -> bool:

        if value == "":
            return True

        if not value.startswith('"'):
            return False

        # A complete JSON string.
        if len(value) >= 2 and value.endswith('"'):

            try:
                result = json.loads(value)

            except json.JSONDecodeError:
                return False

            return isinstance(result, str)

        return self._valid_string_prefix(value)

    @staticmethod
    def _valid_string_prefix(value: str) -> bool:
        """
        Check whether value can still become a valid JSON string.
        """

        if not value.startswith('"'):
            return False

        escaped = False

        for char in value[1:]:

            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                return False

            if ord(char) < 0x20:
                return False

        return True

    def is_complete(self, value: str) -> bool:

        if len(value) < 2:
            return False

        if not value.startswith('"'):
            return False

        if not value.endswith('"'):
            return False

        try:
            result = json.loads(value)

        except json.JSONDecodeError:
            return False

        return isinstance(result, str)


# ============================================================================
# ArrayDecoder
# ============================================================================

class ArrayDecoder(ValueDecoder):
    """
    Decode JSON arrays.

    Examples:

        []
        [1]
        [1, 2]
        ["EUR", "USD"]
        [true, false]
    """

    def is_valid_prefix(self, value: str) -> bool:

        if value == "":
            return True

        if not value.startswith("["):
            return False

        # Complete JSON array.
        try:
            result = json.loads(value)

            return isinstance(result, list)

        except json.JSONDecodeError:
            pass

        return self._balanced_prefix(value)

    @staticmethod
    def _balanced_prefix(value: str) -> bool:

        depth = 0
        in_string = False
        escaped = False

        for char in value:

            if in_string:

                if escaped:
                    escaped = False

                elif char == "\\":
                    escaped = True

                elif char == '"':
                    in_string = False

                continue

            if char == '"':
                in_string = True

            elif char == "[":
                depth += 1

            elif char == "]":
                depth -= 1

                if depth < 0:
                    return False

        if in_string:
            return True

        return depth >= 1

    def is_complete(self, value: str) -> bool:

        if not value:
            return False

        try:
            result = json.loads(value)

        except json.JSONDecodeError:
            return False

        return isinstance(result, list)


# ============================================================================
# ObjectDecoder
# ============================================================================

class ObjectDecoder(ValueDecoder):
    """
    Decode JSON objects.

    Examples:

        {}
        {"a": 1}
        {"currency": "EUR"}
    """

    def is_valid_prefix(self, value: str) -> bool:

        if value == "":
            return True

        if not value.startswith("{"):
            return False

        try:
            result = json.loads(value)

            return isinstance(result, dict)

        except json.JSONDecodeError:
            pass

        return self._balanced_prefix(value)

    @staticmethod
    def _balanced_prefix(value: str) -> bool:

        depth = 0
        in_string = False
        escaped = False

        for char in value:

            if in_string:

                if escaped:
                    escaped = False

                elif char == "\\":
                    escaped = True

                elif char == '"':
                    in_string = False

                continue

            if char == '"':
                in_string = True

            elif char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth < 0:
                    return False

        if in_string:
            return True

        return depth >= 1

    def is_complete(self, value: str) -> bool:

        if not value:
            return False

        try:
            result = json.loads(value)

        except json.JSONDecodeError:
            return False

        return isinstance(result, dict)


# ============================================================================
# Factory
# ============================================================================

class ValueDecoderFactory:
    """
    Create the appropriate value decoder from a parameter type.
    """

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

        decoder_class = cls.DECODERS.get(parameter_type)

        if decoder_class is None:
            raise ValueError(
                f"Unsupported parameter type: {parameter_type}"
            )

        return decoder_class(
            model=model,
            vocabulary=vocabulary,
        )