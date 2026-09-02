import re
from abc import ABC, abstractmethod

COMMON_REGEX_SYNONYMS = {
    "number": r"\d+", "numbers": r"\d+",
    "digit": r"\d", "digits": r"\d",
    "vowel": r"[aeiouAEIOU]", "vowels": r"[aeiouAEIOU]",
    "space": r"\s+", "spaces": r"\s+",
    "punctuation": r"[^\w\s]+",
    # "asterisk": "*", "asterisks": "*",
}

COMMON_REPLACEMENT_SYNONYMS = {
    "asterisk": "*", "asterisks": "*",
    "underscore": "_", "underscores": "_",
    "dash": "-", "dashes": "-",
    "hash": "#", "hashes": "#",
    "star": "*", "stars": "*",
}

def guess_regex_from_prompt(prompt: str) -> str | None:
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    for word in words:
        if word in COMMON_REGEX_SYNONYMS:
            return COMMON_REGEX_SYNONYMS[word]
    return None

# def extract_literal_candidates(prompt: str) -> list[str]:
#     """
#     Extract literal values from the prompt, in order of appearance:
#     quoted spans (single or double), plus a trailing unquoted word/phrase
#     after 'with' (covers '... with asterisks', '... with NUMBERS').
#     """
#     candidates: list[tuple[int, str]] = []

#     for match in re.finditer(r"'([^']*)'|\"([^\"]*)\"", prompt):
#         value = match.group(1) if match.group(1) is not None else match.group(2)
#         candidates.append((match.start(), value))

#     trailing = re.search(r"\bwith\s+([A-Za-z0-9_]+)\s*$", prompt)
#     if trailing and not any(v == trailing.group(1) for _, v in candidates):
#         candidates.append((trailing.start(1), trailing.group(1)))

#     candidates.sort(key=lambda pair: pair[0])
#     return [value for _, value in candidates]


def extract_literal_candidates(prompt: str) -> list[str]:
    candidates: list[tuple[int, str]] = []

    for match in re.finditer(r"'([^']*)'|\"([^\"]*)\"", prompt):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        candidates.append((match.start(), value))

    # Unix (/a/b) or Windows (C:\a\b) file paths
    for match in re.finditer(r"(?:[A-Za-z]:\\[^\s,]+|/[^\s,]+)", prompt):
        value = match.group(0)
        if not any(value == v for _, v in candidates):
            candidates.append((match.start(), value))

    # "with <value> encoding" — capture just the value, not the trailing word
    encoding_match = re.search(r"\bwith\s+([\w.\-]+)\s+encoding\b", prompt, re.IGNORECASE)
    if encoding_match and not any(v == encoding_match.group(1) for _, v in candidates):
        candidates.append((encoding_match.start(1), encoding_match.group(1)))

    trailing = re.search(r"\bwith\s+([A-Za-z0-9_]+)\s*$", prompt)
    if trailing and not any(v == trailing.group(1) for _, v in candidates):
        candidates.append((trailing.start(1), trailing.group(1)))

    candidates.sort(key=lambda pair: pair[0])
    return [value for _, value in candidates]

# def assign_string_targets(context) -> dict[str, str]:
#     """
#     Maps each non-synthesized string parameter to a literal extracted from
#     the prompt. Heuristic: the single LONGEST literal is the "operate on
#     this text" parameter (first such param in the function's signature —
#     e.g. source_string). Any remaining literal is filled from the END of
#     what's left into subsequent params (e.g. replacement), since prompts
#     naturally state '... X ... with Y' — Y is mentioned last. Params that
#     get no match fall back to free generation.
#     """
#     candidates = list(context.literal_candidates)
#     parameter_names = list(context.selected_function.parameters)

#     non_regex_string_params = [
#         name for name in parameter_names
#         if context.selected_function.parameters[name]["type"] == "string"
#         and name.lower() not in ParameterValueFactory.SYNTHESIZED_STRING_PARAMS
#     ]

#     assignment: dict[str, str] = {}
#     if not non_regex_string_params or not candidates:
#         return assignment

#     remaining = candidates[:]

#     primary_param = non_regex_string_params[0]
#     longest_index = max(range(len(remaining)), key=lambda i: len(remaining[i]))
#     assignment[primary_param] = remaining.pop(longest_index)

#     for name in reversed(non_regex_string_params[1:]):
#         if not remaining:
#             break
#         assignment[name] = remaining.pop()

#     return assignment


def extract_colon_suffix(prompt: str) -> str | None:
    """
    Everything after the first ':' in the prompt, trimmed. For single-
    string-param functions (e.g. fn_format_template) the whole remainder —
    including any quotes/braces it contains — IS the literal value, not a
    fragment carved out by quote-matching.
    """
    match = re.search(r":\s*(.+)$", prompt)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def assign_string_targets(context) -> dict[str, str]:
    parameter_names = list(context.selected_function.parameters)

    non_regex_string_params = [
        name for name in parameter_names
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
        # no colon in prompt — fall through to normal candidate logic below

    candidates = list(context.literal_candidates)
    if not candidates:
        return assignment

    remaining = candidates[:]

    primary_param = non_regex_string_params[0]
    longest_index = max(range(len(remaining)), key=lambda i: len(remaining[i]))
    assignment[primary_param] = remaining.pop(longest_index)

    for name in reversed(non_regex_string_params[1:]):
        if not remaining:
            break
        assignment[name] = remaining.pop()

    return assignment

def guess_replacement_from_prompt(prompt: str) -> str | None:
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    for word in words:
        if word in COMMON_REPLACEMENT_SYNONYMS:
            return COMMON_REPLACEMENT_SYNONYMS[word]
    return None

class ParameterValueRule(ABC):

    _token_text_cache: dict[int, str] = {}

    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary
        self.token_ids: list[int] = []

    def _get_token_text(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self.model.decode([token_id])

        return self._token_text_cache[token_id]

    @abstractmethod
    def get_allowed_tokens(self) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def can_finish(self) -> bool:
        raise NotImplementedError

    def validate(self) -> bool:
        return True
    
    def add_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def get_value(self) -> str:
        return self.model.decode(self.token_ids)


class NumberParameterRule(ParameterValueRule):

    def get_value(self) -> str:
        return self.model.decode(self.token_ids).strip()

    def get_allowed_tokens(self) -> set[int]:
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
        """Checks if the value is on a valid path to becoming a number."""
        if value == "" or value == "-":
            return True

        return bool(re.fullmatch(r"-?\d+(?:\.\d*)?", value))

    def can_finish(self) -> bool:
        """Checks if the value is a complete number (can end here)."""
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", self.get_value()))

    def validate(self) -> bool:
        """Validates intermediate tokens during generation."""
        return self._is_valid_prefix(self.get_value())

        
class IntegerParameterRule(ParameterValueRule):

    def get_value(self) -> str:
        return self.model.decode(self.token_ids).strip()

    def get_allowed_tokens(self) -> set[int]:
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
        """Allow empty string or negative sign as initial prefix, followed strictly by digits."""
        if value == "" or value == "-":
            return True

        return bool(re.fullmatch(r"-?\d+", value))

    def can_finish(self) -> bool:
        """Return True only when the accumulated text forms a complete integer."""
        return bool(re.fullmatch(r"-?\d+", self.get_value()))

    def validate(self) -> bool:
        """Validate that the intermediate token sequence can form an integer."""
        return self._is_valid_prefix(self.get_value())


class BooleanParameterRule(ParameterValueRule):

    TRUTHY = {"true", "1", "yes", "y", "on", "True"}
    FALSY = {"false", "0", "no", "n", "off", "False"}
    VALID_VALUES = TRUTHY | FALSY

    def get_value(self) -> str:
        return self.model.decode(self.token_ids).strip()

    def get_allowed_tokens(self) -> set[int]:
        allowed = set()
        current_value = self.get_value()

        # If a complete valid boolean string has been generated, stop accepting further tokens
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
        if value == "":
            return True
        val_lower = value.lower()
        return any(v.lower().startswith(val_lower) for v in self.VALID_VALUES)

    def can_finish(self) -> bool:
        return self.get_value().lower() in {v.lower() for v in self.VALID_VALUES}

    def validate(self) -> bool:
        return self._is_valid_prefix(self.get_value())
    

class StringParameterRule(ParameterValueRule):

    SYNTHESIZE_MAX_LEN = 20  

    def __init__(self, model, vocabulary, target: str | None = None, mode: str = "free"):
        super().__init__(model, vocabulary)
        self.mode = mode
        self.target_tokens = (
            model.encode(target)[0].tolist() if (mode == "exact" and target) else None
        )

    def get_value(self) -> str:
        raw = self.model.decode(self.token_ids).strip()
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            raw = raw[1:-1]
        return raw.strip(' "\'')

    def get_allowed_tokens(self) -> set[int]:
        if self.mode == "exact" and self.target_tokens:
            if len(self.token_ids) < len(self.target_tokens):
                return {self.target_tokens[len(self.token_ids)]}
            return set()

        current_value = self.get_value()

        # Force closure once we've generated a plausible-length pattern —
        # ensures synthesize mode always terminates even if the model
        # never "chooses" to stop on its own.
        if self.mode == "synthesize" and len(current_value) >= self.SYNTHESIZE_MAX_LEN:
            return set()   # no more value chars allowed; can_finish() lets the caller close out

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
        if self.mode == "exact" and self.target_tokens:
            return self.token_ids == self.target_tokens
        return bool(self.get_value())

    def validate(self) -> bool:
        if self.mode == "exact" and self.target_tokens:
            return self.token_ids == self.target_tokens[: len(self.token_ids)]
        val = self.get_value()
        return not any(char in val for char in [",", ")", "(", ":", "\n"])

class ParameterValueFactory:

    SYNTHESIZED_STRING_PARAMS = {"regex", "pattern"}

    @staticmethod
    def build(parameter_type, model, vocabulary, target: str | None = None, parameter_name: str = ""):

        if parameter_type == "number":
            return NumberParameterRule(model, vocabulary)

        if parameter_type == "string":
            # if parameter_name.lower() in ParameterValueFactory.SYNTHESIZED_STRING_PARAMS:
            #     return StringParameterRule(model, vocabulary, mode="synthesize")
            # if target is not None:
            #     return StringParameterRule(model, vocabulary, target=target, mode="exact")
            # return StringParameterRule(model, vocabulary, mode="free")
            if target is not None:
                # Works whether target came from the regex dictionary
                # (guess_regex_from_prompt) or from literal-copy assignment
                # (assign_string_targets) — either way it's a known-correct
                # value, so lock to it exactly.
                return StringParameterRule(model, vocabulary, target=target, mode="exact")
            if parameter_name.lower() in ParameterValueFactory.SYNTHESIZED_STRING_PARAMS:
                # No dictionary hit (e.g. 'cat') — let the model synthesize,
                # permitting regex syntax chars.
                return StringParameterRule(model, vocabulary, mode="synthesize")
            return StringParameterRule(model, vocabulary, mode="free")

        if parameter_type == "integer":
            return IntegerParameterRule(model, vocabulary)

        if parameter_type == "boolean":
            return BooleanParameterRule(model, vocabulary)

        raise ValueError(f"Unsupported parameter type: {parameter_type}")