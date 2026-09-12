import re
from abc import ABC, abstractmethod
from typing import cast

from llm_sdk import Small_LLM_Model
from .data_models import GenerationContext

COMMON_REGEX_SYNONYMS = {
    "number": r"\d+",
    "numbers": r"\d+",
    "digit": r"\d",
    "digits": r"\d",
    "vowel": r"[aeiouAEIOU]",
    "vowels": r"[aeiouAEIOU]",
    "space": r"\s+",
    "spaces": r"\s+",
    "punctuation": r"[^\w\s]+",
    # "asterisk": "*", "asterisks": "*",
}

COMMON_REPLACEMENT_SYNONYMS = {
    "asterisk": "*",
    "asterisks": "*",
    "underscore": "_",
    "underscores": "_",
    "dash": "-",
    "dashes": "-",
    "hash": "#",
    "hashes": "#",
    "star": "*",
    "stars": "*",
}


def guess_regex_from_prompt(prompt: str) -> str | None:
    """Return a regex pattern based on a keyword found in the prompt."""
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    for word in words:
        if word in COMMON_REGEX_SYNONYMS:
            return COMMON_REGEX_SYNONYMS[word]
    return None


def extract_literal_candidates(prompt: str) -> list[str]:
    """Extract literal values such as quoted text, file paths,
    and trailing values from the prompt."""
    candidates: list[tuple[int, str]] = []

    for match in re.finditer(r"'([^']*)'|\"([^\"]*)\"", prompt):
        value = match.group(1) \
            if match.group(1) is not None else match.group(2)
        candidates.append((match.start(), value))

    # Unix (/a/b) or Windows (C:\a\b) file paths
    for match in re.finditer(r"(?:[A-Za-z]:\\[^\s,]+|/[^\s,]+)", prompt):
        value = match.group(0)
        if not any(value == v for _, v in candidates):
            candidates.append((match.start(), value))

    encoding_match = re.search(
        r"\bwith\s+([\w.\-]+)\s+encoding\b", prompt, re.IGNORECASE
    )
    if (
        encoding_match and
        not any(v == encoding_match.group(1)
                for _, v in candidates)
    ):
        candidates.append(
            (encoding_match.start(1), encoding_match.group(1)))

    trailing = re.search(r"\bwith\s+([A-Za-z0-9_]+)\s*$", prompt)
    if trailing and not any(v == trailing.group(1) for _, v in candidates):
        candidates.append((trailing.start(1), trailing.group(1)))

    candidates.sort(key=lambda pair: pair[0])
    return [value for _, value in candidates]


def extract_trailing_word(prompt: str) -> str | None:
    """Return the last word in the prompt, stripped of trailing
    punctuation. Covers bare-name prompts with no quotes, no 'with',
    and no colon — e.g. 'Greet shrek', 'Greet john'."""
    match = re.search(r"([A-Za-z0-9_]+)[.!?]*\s*$", prompt.strip())
    return match.group(1) if match else None


def extract_colon_suffix(prompt: str) -> str | None:
    """Return the text appearing after the first colon in the prompt."""
    match = re.search(r":\s*(.+)$", prompt)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def assign_string_targets(
        context: GenerationContext
) -> dict[str, str]:
    """Assign extracted literal values to the string
    parameters of the selected function."""
    if context.selected_function is None:
        raise RuntimeError("No function selected")

    parameter_names: list[str] = list(context.selected_function.parameters)

    non_regex_string_params: list[str] = [
        name
        for name in parameter_names
        if context.selected_function.parameters[name]["type"] == "string"
        and name.lower() not in ParameterValueFactory.SYNTHESIZED_STRING_PARAMS
    ]

    assignment: dict[str, str] = {}
    if not non_regex_string_params:
        return assignment

    if len(non_regex_string_params) == 1:
        only_param = non_regex_string_params[0]
        colon_value = extract_colon_suffix(context.user_prompt)
        if colon_value:
            assignment[only_param] = colon_value
            return assignment
        trailing_word = extract_trailing_word(context.user_prompt)
        if trailing_word:
            assignment[only_param] = trailing_word
            return assignment

    candidates = list(context.literal_candidates)
    if not candidates:
        return assignment

    remaining = candidates[:]

    primary_param = non_regex_string_params[0]
    longest_index = max(range(len(remaining)),
                        key=lambda i: len(remaining[i]))
    assignment[primary_param] = remaining.pop(longest_index)

    for name in reversed(non_regex_string_params[1:]):
        if not remaining:
            break
        assignment[name] = remaining.pop()

    return assignment


def guess_replacement_from_prompt(prompt: str) -> str | None:
    """Return a replacement character
    based on a keyword found in the prompt."""
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    for word in words:
        if word in COMMON_REPLACEMENT_SYNONYMS:
            return COMMON_REPLACEMENT_SYNONYMS[word]
    return None


def guess_regex_target(
    context: GenerationContext,
    assigned: dict[str, str] | None = None,
) -> str | None:
    """Return a regex pattern matching a keyword found in the prompt."""
    category_match = guess_regex_from_prompt(context.user_prompt)
    if category_match:
        return category_match

    if assigned is None:
        assigned = assign_string_targets(context)

    used_values = set(assigned.values())

    for candidate in context.literal_candidates:
        if candidate not in used_values:
            return candidate

    return None


class ParameterValueRule(ABC):
    """Base class for rules that constrain and
    validate parameter values."""

    _token_text_cache: dict[int, str] = {}

    def __init__(
            self, model: Small_LLM_Model,
            vocabulary: list[str]) -> None:
        """Initialize the parameter value rule
        with the model and vocabulary."""
        self.model = model
        self.vocabulary = vocabulary
        self.token_ids: list[int] = []

    def _get_token_text(self, token_id: int) -> str:
        """Return the text represented by a token ID,
        using a cache when available."""
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = \
                self.model.decode([token_id])

        return self._token_text_cache[token_id]

    @abstractmethod
    def get_allowed_tokens(self) -> set[int]:
        """Return the token IDs allowed for the current parameter value."""
        raise NotImplementedError

    @abstractmethod
    def can_finish(self) -> bool:
        """Return whether the current value is complete and valid."""
        raise NotImplementedError

    @abstractmethod
    def _is_valid_prefix(self, value: str) -> bool:
        """Return whether the value can still become valid."""
        raise NotImplementedError

    def validate(self) -> bool:
        """Return whether the current parameter value is a valid prefix."""
        return self._is_valid_prefix(self.get_value())

    def add_token(self, token_id: int) -> None:
        """Add a token to the current parameter value."""
        self.token_ids.append(token_id)

    def get_value(self) -> str:
        """Return the current parameter value as text."""
        return cast(str, self.model.decode(self.token_ids)).strip()


class NumberParameterRule(ParameterValueRule):
    """Constrains parameter values to valid decimal numbers."""

    def get_value(self) -> str:
        """Return the current number value as text."""
        return cast(str, self.model.decode(self.token_ids)).strip()

    def get_allowed_tokens(self) -> set[int]:
        """Return tokens that keep the value a valid number prefix."""
        allowed = set()
        current_value = self.get_value()

        for token_id in range(len(self.vocabulary)):
            # token_text = self.model.decode([token_id])
            token_text = self._get_token_text(token_id)

            if not token_text:
                continue

            candidate = current_value + token_text.strip()

            if self._is_valid_prefix(candidate):
                allowed.add(token_id)

        return allowed

    def _is_valid_prefix(self, value: str) -> bool:
        """Return whether the value can become a valid number."""
        if value == "" or value == "-":
            return True

        return bool(re.fullmatch(r"-?\d+(?:\.\d*)?", value))

    def can_finish(self) -> bool:
        """Checks if the value is a complete number (can end here)."""
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", self.get_value()))


class IntegerParameterRule(ParameterValueRule):
    """Constrains parameter values to valid integers."""

    def get_value(self) -> str:
        """Return the current integer value as text."""
        return cast(str, self.model.decode(self.token_ids)).strip()

    def get_allowed_tokens(self) -> set[int]:
        """Return tokens that keep the value a valid integer prefix."""
        allowed = set()
        current_value = self.get_value()

        for token_id in range(len(self.vocabulary)):
            token_text = self._get_token_text(token_id)

            if not token_text:
                continue

            candidate = current_value + token_text.strip()

            if self._is_valid_prefix(candidate):
                allowed.add(token_id)

        return allowed

    def _is_valid_prefix(self, value: str) -> bool:
        """Return whether the value can become a valid integer.

        Allow empty string or negative sign as initial prefix,
        followed strictly by digits.
        """
        if value == "" or value == "-":
            return True

        return bool(re.fullmatch(r"-?\d+", value))

    def can_finish(self) -> bool:
        """Return whether the current value is a complete integer."""
        return bool(re.fullmatch(r"-?\d+", self.get_value()))


class BooleanParameterRule(ParameterValueRule):
    """Constrains parameter values
    to supported boolean values."""

    TRUTHY = {"true", "1", "yes", "y", "on", "True"}
    FALSY = {"false", "0", "no", "n", "off", "False"}
    VALID_VALUES = TRUTHY | FALSY

    def get_value(self) -> str:
        """Return the current boolean value as text."""
        return cast(str, self.model.decode(self.token_ids)).strip()

    def get_allowed_tokens(self) -> set[int]:
        """Return tokens that keep the value a valid boolean prefix."""

        allowed = set()
        current_value = self.get_value()

        if current_value.lower() in {v.lower() for v in self.VALID_VALUES}:
            return set()

        for token_id in range(len(self.vocabulary)):
            token_text = self._get_token_text(token_id)

            if not token_text:
                continue

            candidate = current_value + token_text.strip()

            if self._is_valid_prefix(candidate):
                allowed.add(token_id)

        return allowed

    def _is_valid_prefix(self, value: str) -> bool:
        """Return whether the value can become a valid boolean."""
        if value == "":
            return True
        val_lower = value.lower()
        return any(v.lower().startswith(val_lower) for v in self.VALID_VALUES)

    def can_finish(self) -> bool:
        """Return whether the current value is a complete boolean."""
        return self.get_value().lower() in \
            {v.lower() for v in self.VALID_VALUES}


class StringParameterRule(ParameterValueRule):
    """Constrains string parameter values using
    free, exact, or synthesized modes."""

    SYNTHESIZE_MAX_LEN = 20

    def __init__(
        self, model: Small_LLM_Model,
        vocabulary: list[str],
        target: str | None = None,
        mode: str = "free"
    ):
        """Initialize the string rule with an optional target
        and generation mode."""
        super().__init__(model, vocabulary)
        self.mode = mode
        self.target_tokens = (
            cast(list[int], model.encode(target)[0].tolist())
            if (mode == "exact" and target) else None
        )

    def get_value(self) -> str:
        """Return the current string value as text."""
        raw: str = cast(
            str,
            self.model.decode(self.token_ids),).strip()
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            raw = raw[1:-1]
        return raw.strip(" \"'")

    def get_allowed_tokens(self) -> set[int]:
        """Return tokens allowed by the current string generation mode."""
        if self.mode == "exact" and self.target_tokens:
            if len(self.token_ids) < len(self.target_tokens):
                return {self.target_tokens[len(self.token_ids)]}
            return set()

        current_value = self.get_value()

        if (
            self.mode == "synthesize"
            and len(current_value) >= self.SYNTHESIZE_MAX_LEN
        ):
            return (set())

        allowed = set()
        depth = current_value.count("(") - current_value.count(")")

        for token_id in range(len(self.vocabulary)):
            token_text = self._get_token_text(token_id)
            if not token_text:
                continue
            if any(char in token_text for char in ["\n", "\r"]):
                continue
            if '"' in token_text or "'" in token_text:
                continue
            if not current_value and token_text.startswith(" "):
                continue

            if self.mode == "synthesize":
                if ":" in token_text:
                    continue
                if "," in token_text and depth <= 0:
                    continue
                if ")" in token_text and depth <= 0 and "(" not in token_text:
                    continue
            else:
                if any(char in token_text for char in [",", ")", "(", ":"]):
                    continue

            allowed.add(token_id)

        return allowed

    def can_finish(self) -> bool:
        """Return whether the current string value can be completed."""
        if self.mode == "exact" and self.target_tokens:
            return self.token_ids == self.target_tokens
        return bool(self.get_value())

    def _is_valid_prefix(self, value: str) -> bool:
        """Return whether the current string value is valid."""
        if self.mode == "exact" and self.target_tokens:
            return self.token_ids == self.target_tokens[: len(self.token_ids)]

        return not any(char in value for char in [",", ")", "(", ":", "\n"])


class ParameterValueFactory:
    """Creates the appropriate parameter-value rule for a parameter type."""

    SYNTHESIZED_STRING_PARAMS = {"regex"}

    @staticmethod
    def build(
        parameter_type: str,
        model: Small_LLM_Model,
        vocabulary: list[str],
        target: str | None = None,
        parameter_name: str = "",
    ) -> ParameterValueRule:
        """Create a parameter value rule
        based on the parameter type and target."""

        if parameter_type == "number":
            return NumberParameterRule(model, vocabulary)

        if parameter_type == "string":
            if target is not None:
                return StringParameterRule(
                    model, vocabulary, target=target, mode="exact"
                )
            if (
                parameter_name.lower()
                in ParameterValueFactory.SYNTHESIZED_STRING_PARAMS
            ):
                return StringParameterRule(
                    model, vocabulary, mode="synthesize")
            return StringParameterRule(model, vocabulary, mode="free")

        if parameter_type == "integer":
            return IntegerParameterRule(model, vocabulary)

        if parameter_type == "boolean":
            return BooleanParameterRule(model, vocabulary)

        raise ValueError(f"Unsupported parameter type: {parameter_type}")
