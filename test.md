*This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].*

# call me maybe — Function Calling with Constrained Decoding

## Description

This project translates natural-language prompts into structured, machine-executable
function calls — the same problem modern AI assistants solve when deciding "which API
do I call, with which arguments?" instead of just answering in prose.

Given a prompt like:

```
Reverse the string 'world'
```

the goal is not to have the model *answer* the question, but to have it produce:

```json
{
  "name": "fn_reverse_string",
  "parameters": { "s": "world" }
}
```

The twist is the model: **Qwen/Qwen3-0.6B**, a 500M-parameter model that is, on its
own, notoriously unreliable at producing valid structured output — free-form prompting
alone succeeds well under 100% of the time. Rather than hoping the model gets the JSON
right, this project uses **constrained decoding**: at every generation step, the set of
tokens the model is allowed to pick from is restricted to only those that keep the output
both syntactically valid and schema-compliant. Invalid tokens are simply never offered
to the model, so the output is guaranteed to parse and match the function signature —
100% of the time, by construction, not by luck.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package, copied alongside `src/` (not included in this repo per subject
  rules — copy it from the provided package)

### Installation

```bash
make install
```

This runs `uv sync`, installing `numpy` and `pydantic` (the only two third-party
dependencies allowed by the subject) into a project-local virtual environment.

### Running

```bash
make run
```

which is equivalent to:

```bash
uv run python -m src
```

By default, the program reads from `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes to
`data/output/function_calling_results.json`. Custom paths can be supplied:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other Makefile targets

| Target | Purpose |
|---|---|
| `make install` | Install dependencies via `uv sync` |
| `make run` | Run the main script |
| `make debug` | Run the main script under `pdb` |
| `make lint` | `flake8` + `mypy` with the subject-mandated flags |
| `make lint-strict` | `flake8` + `mypy --strict` |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, and similar caches |

## Algorithm Explanation

### Overview

Generation proceeds token-by-token through a small **state machine**
(`generation_state.py`), where each state represents "what part of the function call
we're currently writing" and exposes exactly one method that matters:
`get_allowed_tokens()` — the set of token IDs the model is permitted to emit next.
At every step:

1. The model produces logits for the *entire* vocabulary via `get_logits_from_input_ids`.
2. The current state's `get_allowed_tokens()` restricts consideration to a small
   legal subset.
3. The token with the highest logit *among the allowed set* is picked
   (`highest_allowed_tokens`) — everything outside the allowed set is treated as if
   its logit were `-inf`, without actually needing to touch the full logits array.
4. The chosen token is appended, the state observes it, and transitions to the next
   state once its local grammar is satisfied.

The states, in order, are:

```
FunctionNameState → ParametersState "(" → ParameterNameState
  → ParameterValueSeparatorState ":" → ParameterValueState
  → [ParameterNameState for the next parameter, or termination] ")"
```

**`FunctionNameState`** constrains generation to only the token sequences that spell out
one of the known function names (a trie-style prefix walk over
`function_token_sequences`), so the model can never invent a function that doesn't
exist.

**`ParameterNameState`** does the same for the current function's next expected
parameter name — the model can't misspell or reorder parameters.

**`ParameterValueState`** is where type-specific rules apply, delegated to
`parameter_values.py`:

- **`number` / `integer`**: allowed tokens are restricted to those that keep the
  accumulated text a valid prefix of a number (`-?\d+(\.\d*)?` for `number`,
  `-?\d+` for `integer`), so malformed numerics are structurally impossible.
- **`boolean`**: allowed tokens are restricted to prefixes of a fixed truthy/falsy
  vocabulary (`true`, `false`, `1`, `0`, `yes`, `no`, ...).
- **`string`**: the most interesting case (see below).

### String parameters: three modes

A naive "allow any non-structural character" rule for strings is not enough — it
guarantees *syntactic* validity but not *semantic* correctness, and with a model this
small, unconstrained string generation tends to hallucinate (e.g. writing the parameter's
*type* instead of its *value*, or copying a random nearby word). `StringParameterRule`
therefore operates in one of three modes, chosen per-parameter by
`ParameterValueState`:

- **`exact`**: the value is already known (extracted from the prompt or resolved via a
  synonym dictionary — see below), so generation is forced token-by-token to reproduce
  that exact string. This is O(1) per step and cannot drift, truncate early, or
  hallucinate, since the model isn't actually choosing anything.
- **`synthesize`**: used when a value must be *computed* rather than copied (a regex
  pattern with no dictionary match, e.g. matching a literal word). Free generation is
  allowed, but structural characters needed for real regex syntax (`(`, `)`, `,`) are
  permitted rather than blocked, and a length cap forces termination so generation can't
  run away.
- **`free`**: a plain fallback for string parameters with no extractable target and no
  synthesis requirement — free generation with the standard structural-character
  blocklist (`,` `)` `(` `:` etc., which would otherwise be misread as part of the
  function-call syntax itself).

### Extracting values from the prompt

Before generation starts, `extract_literal_candidates` scans the prompt for
quoted spans (`'...'` / `"..."`), file-path-shaped substrings, and a couple of common
phrasing patterns (`with X encoding`, trailing `with X`). `assign_string_targets` then
maps each of the function's string parameters to a candidate: the longest literal
becomes the "operate on this" parameter (e.g. `source_string`), and remaining literals
fill subsequent parameters (e.g. `replacement`) in prompt order. For single-string-
parameter functions, the entire prompt content after the first `:` is used verbatim, so
that quotes, apostrophes, and braces embedded in the target value are preserved rather
than mangled by quote-boundary slicing.

For `regex`/`pattern` parameters specifically, a small synonym dictionary
(`COMMON_REGEX_SYNONYMS`) maps common category words to real regex ("numbers" → `\d+`,
"vowels" → `[aeiouAEIOU]`, etc.) and is checked before falling back to `synthesize` mode
— this is deliberately a lookup, not left to model reasoning, because a 500M model
cannot reliably derive correct regex syntax from a description alone, even with a
few-shot example in the prompt.

## Design Decisions

- **Grammar-first, not prompt-first.** The subject is explicit that hoping a small model
  spontaneously emits valid JSON is not acceptable, and our own testing confirmed this:
  free-form generation drifted into type names, truncated strings early, or echoed
  unrelated prompt tokens. Every syntactic and type guarantee in this project is enforced
  by restricting the *allowed token set*, never by asking nicely.
- **Dictionaries over model reasoning for finite categories.** Regex synonyms and
  symbolic replacements (`"asterisks"` → `*`) are small, enumerable categories where a
  lookup table is strictly more reliable than hoping a 500M model infers the mapping —
  constrained decoding guarantees valid *syntax*, but not semantic correctness, and that
  gap has to be closed with domain knowledge, not more grammar.
- **Vocabulary decode caching.** `model.decode([token_id])` is expensive when called
  per-candidate-token per-generation-step (full vocab scans compounded over dozens of
  steps). A shared, class-level cache (`_token_text_cache`) memoizes each token's decoded
  text the first time it's seen, turning what was originally a multi-minute generation
  into a near-instant one.
- **A hard `max_tokens` cap.** Even with grammar constraints, `synthesize`/`free` modes
  have no guarantee the model will *choose* to emit a terminating token. `generate_tokens`
  caps total generated length and raises a descriptive `RuntimeError` (including the
  offending state and partial output) rather than allowing an unbounded loop.
- **Pydantic for all shared state.** Per the subject's requirements, `Function_definition`
  and `GenerationContext` are pydantic models, giving free validation and an explicit,
  typed contract for what data flows through the generation pipeline.

## Performance Analysis

- **Correctness**: on the provided and extended test prompts (arithmetic, greetings,
  string reversal, square roots, regex substitution with both quoted-literal and
  synonym-category patterns, file paths, and template formatting), the constrained
  decoder produces syntactically valid, schema-compliant JSON on every run — this is a
  structural guarantee, not a measured success rate, since invalid tokens are never
  offered to the model in the first place.
- **Speed**: after introducing the token-decode cache, per-prompt generation is fast —
  dominated by the number of forward passes through the model (one per generated token),
  not by Python-side vocabulary scanning.
- **Reliability under edge cases**: empty candidate extraction, prompts with no quotes,
  and functions with a single string parameter are all handled via fallback paths
  (`free` mode, colon-suffix extraction) rather than crashing or hanging.

## Challenges Faced

- **String value hallucination.** Early on, unconstrained string generation produced
  values like `"String"` (the model completing `s:` as if writing a type annotation)
  instead of the actual prompt content. Fixed by forcing string parameters into `exact`
  mode against a target extracted from the prompt whenever one is available.
- **An inverted space-blocking condition** in the original string rule prevented
  multi-word values from ever being generated past the first word. Fixed by only
  blocking a leading space at the very *start* of the value, not after content already
  exists.
- **Naive substring matching** (`candidate in prompt_text`) produced false-positive
  partial matches (`"str"` matching inside `"string"`) and allowed premature
  termination the instant *any* non-empty match existed. Fixed by switching to exact,
  token-by-token target matching instead of substring containment checks.
- **A silent performance cliff.** Re-decoding the full vocabulary on every character of
  every generation step made longer prompts (multi-word strings, synthesized regex)
  look like they had hung, when they were simply doing tens of thousands of redundant
  decode calls. Fixed with the token-text cache described above.
- **Regex semantics vs. syntax.** Constrained decoding guaranteed the *regex* parameter
  was syntactically valid, but the model still had no reliable way to know that
  "numbers" means `\d+` — a few-shot example helped partially but wasn't robust (the
  model sometimes copied a salient nearby word instead). Resolved with an explicit
  synonym dictionary as the primary path, with model synthesis retained only as a
  fallback for genuinely open-ended cases.

## Testing Strategy

- Manual testing against the subject's example prompts (`fn_add_numbers`, `fn_greet`,
  `fn_reverse_string`) as a baseline.
- Adversarial prompt design targeting each string-handling edge case as it was found:
  multi-word literals, embedded quotes/apostrophes/braces, Windows and Unix file paths,
  regex-requiring functions with both literal and category-based patterns, and
  single-vs-multi string-parameter functions.
- Re-running the full prompt set after each fix to confirm no regressions on previously
  passing cases.
- (Not submitted, per subject rules) `pytest`/`unittest` cases exercising
  `extract_literal_candidates`, `assign_string_targets`, `guess_regex_from_prompt`, and
  the constrained-token-set logic in isolation from the model.

## Example Usage

```bash
$ uv run python -m src
Loading weights: 100%|████████████████████| 290/290
...
User prompt:
Reverse the string 'world'

Function call:
fn_reverse_string(s:world)
{
    "prompt": "Reverse the string 'world'",
    "name": "fn_reverse_string",
    "parameters": { "s": "world" }
}
```

```bash
$ uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

## Resources

- [Anthropic — Tool use / function calling overview](https://docs.claude.com/en/docs/build-with-claude/tool-use)
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [Hugging Face — `outlines` and grammar-constrained generation (conceptual reference,
  not used as a dependency per subject rules)](https://github.com/dottxt-ai/outlines)
- [Byte-Pair Encoding tokenization overview](https://huggingface.co/learn/nlp-course/chapter6/5)
- Python `re` module documentation, for the regex patterns used in literal/pattern
  extraction

### AI usage disclosure

AI assistance (Claude) was used throughout development for:

- Diagnosing the root causes of several generation bugs (string hallucination,
  premature termination from substring-matching, an inverted whitespace condition, and
  an unbounded-generation hang caused by uncached vocabulary decoding).
- Proposing and iterating on the constrained-decoding design for string parameters
  (`exact` / `synthesize` / `free` modes), including the literal-extraction heuristics
  and the regex/replacement synonym dictionaries.
- Reviewing the codebase for dead code and redundant logic during cleanup.
- Drafting this README.

All AI-suggested code was read, tested against the project's actual prompts, and
adjusted based on real output before being kept — nothing was copied in without
understanding what it does or verifying it against the subject's requirements.



Support for multiple models, tested models include

Qwen/Qwen2.5-0.5B-Instruct
HuggingFaceTB/SmolLM2-360M-Instruct
Qwen/Qwen2.5-1.5B-Instruct
HuggingFaceTB/SmolLM2-1.7B-Instruct