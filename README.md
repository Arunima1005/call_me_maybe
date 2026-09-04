*This project has been created as part of the 42 curriculum by <your-login>.*

# Call Me Maybe

## Description

**Call Me Maybe** is a project about function calling with Large Language Models (LLMs).

The goal is to build a function-calling tool that translates a natural-language user request into a structured function call.

For example, given:

```text
What is the sum of 40 and 2?
```

the program should identify the appropriate function and extract its arguments:

```json
{
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 40.0,
        "b": 2.0
    }
}
```

The main challenge is that small language models are not always reliable at producing structured output. Instead of relying only on prompting the model to produce valid JSON, this project uses **constrained decoding** to control the generated tokens.

The decoder restricts the model to tokens that are valid for the current generation state and parameter type. This makes it possible to generate a structurally valid function call while still using a small language model.

---

## Project Goals

The implementation has four main goals:

* Select the correct function using the LLM.
* Extract the correct values from the user's natural-language prompt.
* Constrain token generation according to the expected function-call structure.
* Produce valid JSON with correctly typed parameters.

The implementation is designed to work with different function definitions and test prompts rather than hardcoding the examples provided by the subject.

---

## How It Works

The overall generation pipeline is:

```text
User Prompt
     |
     v
Build Prompt
     |
     v
Tokenization
     |
     v
Input Token IDs
     |
     v
LLM
     |
     v
Logits for Next Token
     |
     v
Constrained Token Selection
     |
     v
Generation State
     |
     +---- Function Name
     |
     +---- "("
     |
     +---- Parameter Name
     |
     +---- ":"
     |
     +---- Parameter Value
     |
     +---- "," / ")"
     |
     v
Structured Function Call
     |
     v
JSON Output
```

Each generated token is selected individually. At every step, the current state determines which tokens are allowed.

---

# Installation

## Requirements

* Python 3.10 or later
* `uv`
* The provided `llm_sdk` package
* `numpy`
* `pydantic`

The project uses the provided `Small_LLM_Model` wrapper to interact with the language model.

The subject requires Python 3.10+, type hints, PEP 257 docstrings, flake8 compliance, mypy checking, and graceful error handling.

## Install dependencies

Create/synchronize the environment with:

```bash
uv sync
```

The project can also be installed using the Makefile:

```bash
make install
```

The reviewer can use `uv sync` to install the project dependencies.

---

# Project Structure

```text
.
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
│
├── llm_sdk/
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── data_models.py
│   ├── generation_state.py
│   └── parameter_values.py
│
├── Makefile
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

The subject requires the repository to contain `src/`, `pyproject.toml`, `uv.lock`, `llm_sdk/`, `data/input/`, and `README.md`. The generated `output/` directory should not be committed.

---

# Input Files

The program uses two JSON input files.

## `functions_definition.json`

This file describes the functions that are available to the LLM.

Each function contains:

* `name`
* `description`
* `parameters`
* `returns`

Example:

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    }
]
```

The subject specifies that the function definitions contain the function name, argument names/types, return type, and description.

## `function_calling_tests.json`

This file contains the natural-language prompts that the program processes.

Example:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    },
    {
        "prompt": "Reverse the string 'hello'"
    }
]
```

The subject states that the input files may change during peer review, so the implementation must not rely on hardcoded examples.

---

# Usage

## Default execution

The program can be executed with:

```bash
uv run python -m src
```

The default paths are:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

These defaults are configured through `argparse`.

## Specify input and output files

```bash
uv run python -m src \
    --functions data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

Short options are also available:

```bash
uv run python -m src \
    -f data/input/functions_definition.json \
    -i data/input/function_calling_tests.json \
    -o data/output/function_calling_results.json
```

## Select the LLM model

The implementation also supports selecting the model using `-m` / `--model`.

Default:

```bash
uv run python -m src
```

uses:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

A different supported model can be selected with:

```bash
uv run python -m src \
    --model "HuggingFaceTB/SmolLM2-360M-Instruct"
```

For example:

```bash
uv run python -m src \
    --model "Qwen/Qwen2.5-1.5B-Instruct"
```

The model is passed to:

```python
Small_LLM_Model(args.model)
```

so the rest of the constrained-decoding implementation does not need to change.

The subject allows other models as long as the required default model remains supported.

---

# Makefile

The Makefile provides commands for common project operations.

Typical commands are:

```bash
make install
make run
make lint
make debug
make clean
```

The `run` target executes the project using:

```bash
uv run python -m src
```

The subject requires the Makefile to provide `install`, `run`, `debug`, `clean`, and `lint` targets.

---

# Constrained Decoding Algorithm

## 1. Build the prompt

For every user request, the program builds a prompt containing:

* Instructions for function selection.
* The available function definitions.
* The user's natural-language request.
* An indication that a function call should be generated.

The prompt instructs the LLM to select the best function and preserve the exact values mentioned by the user.

Each test prompt is processed independently.

---

## 2. Tokenization

The prompt is encoded using the model tokenizer:

```python
input_ids = model.encode(function_prompt)[0].tolist()
```

The vocabulary is also loaded from the vocabulary file provided by the SDK.

The vocabulary provides the mapping between token IDs and their text representation.

The subject specifically requires the vocabulary to be used to determine which tokens are valid during constrained decoding.

---

## 3. Function Selection

The first generation state is:

```text
FunctionNameState
```

The available function names are tokenized in advance.

The state only allows tokens that are prefixes of one of the available function names.

For example, if the available functions are:

```text
fn_add_numbers
fn_greet
fn_reverse_string
```

the decoder cannot generate an arbitrary function name.

When the complete function name has been generated, the corresponding function is stored in the generation context.

This means function selection is performed by the LLM, while constrained decoding guarantees that the generated name belongs to the available function definitions.

---

# Generation State Machine

After selecting the function, generation proceeds through several states:

```text
FunctionNameState
       |
       v
ParametersState
       |
       v
ParameterNameState
       |
       v
ParameterValueSeparatorState
       |
       v
ParameterValueState
       |
       +---- more parameters ----> ParameterNameState
       |
       +---- last parameter ------> End
```

## `GenerationState`

`GenerationState` is the base class for the state machine.

Each state is responsible for:

* Determining allowed tokens.
* Storing generated tokens.
* Detecting completion.
* Moving to the next state.

The shared `GenerationContext` contains information needed throughout generation, including:

* Function definitions.
* Tokenized function names.
* The model.
* Vocabulary.
* Special token IDs.
* Selected function.
* Current parameter index.
* Extracted parameter values.
* User prompt.

---

# Parameter Name Generation

`ParameterNameState` generates the name of the current parameter.

The parameter names are taken directly from the selected function definition.

For example:

```json
"parameters": {
    "a": {
        "type": "number"
    },
    "b": {
        "type": "number"
    }
}
```

results in the parameter sequence:

```text
a
b
```

Only tokens that continue the expected parameter name are allowed.

This prevents the decoder from generating unknown parameter names.

---

# Parameter Value Generation

Once a parameter name has been generated, the decoder generates its value.

The implementation uses `ParameterValueFactory` to select a parameter-specific rule.

Different types have different constraints.

Supported parameter types include:

```text
number
integer
boolean
string
array
object
```

Values are converted to their correct Python/JSON types before the final output is written.

For example:

```python
number -> float
integer -> int
boolean -> bool
string -> str
array -> json.loads(...)
object -> json.loads(...)
```

---

# Number Constraints

`NumberParameterRule` restricts generation to valid decimal number prefixes.

For example, these can be valid intermediate states:

```text
-
12
12.
12.5
-42
-42.75
```

The distinction between a **valid prefix** and a **complete value** is important.

For example:

```text
12.
```

can still become:

```text
12.5
```

but it is not yet a complete number.

The rule therefore provides:

```python
_is_valid_prefix()
```

for intermediate generation and:

```python
can_finish()
```

for determining whether the value is complete.

---

# Integer Constraints

`IntegerParameterRule` only permits integer values.

Examples:

```text
0
42
-10
265
```

A decimal value such as:

```text
10.5
```

cannot be generated as an integer.

Negative numbers are supported, which is important because the generation prompt explicitly instructs the model to preserve negative signs.

---

# Boolean Constraints

`BooleanParameterRule` restricts generation to supported boolean representations.

Examples include:

```text
true
false
1
0
yes
no
y
n
on
off
```

The rule checks whether the generated text is still a valid prefix of one of the supported boolean values.

Once a complete boolean value has been generated, the value can be terminated.

---

# String Constraints

`StringParameterRule` handles string parameters.

The implementation supports different modes:

### Free mode

The model can generate a string while preventing structural characters from accidentally terminating or corrupting the function call.

### Exact mode

When the user's prompt contains a value that should be reproduced exactly, the target string is tokenized first.

The decoder then allows only the next token from that target sequence.

This is useful for requests such as:

```text
Reverse the string 'hello'
```

where the value:

```text
hello
```

should be preserved rather than invented by the model.

### Synthesize mode

Some string parameters need to be generated from the user's request.

The implementation applies additional restrictions to avoid generating structural characters such as commas, colons, or closing parentheses in the wrong place.

---

# Extracting Values From the User Prompt

The implementation also performs lightweight extraction of literal values from the user request.

`extract_literal_candidates()` looks for values such as:

* Quoted strings.
* Unix paths.
* Windows paths.
* Encoding values.
* Values following certain textual patterns.

For example:

```text
Reverse the string 'hello'
```

can provide:

```text
hello
```

as a candidate value.

For functions with string parameters, `assign_string_targets()` associates extracted values with the appropriate parameters.

This extraction does **not** select the function. Function selection remains the responsibility of the LLM, as required by the subject.

---

# Special String Handling

The implementation also handles common natural-language descriptions of values.

For example, regular-expression style parameters can recognise words such as:

```text
number
numbers
digit
digits
vowel
vowels
space
spaces
punctuation
```

and map them to corresponding regular expressions.

Similarly, replacement descriptions such as:

```text
asterisk
underscore
dash
hash
star
```

can be mapped to their corresponding replacement characters.

This allows the generated function arguments to better reflect the user's natural-language request.

---

# Token Selection

At every generation step, the state returns a set of allowed token IDs:

```python
allowed_tokens = state.get_allowed()
```

The model then calculates logits for the current input:

```python
logits = model.get_logits_from_input_ids(input_ids)
```

Instead of selecting the highest logit across the entire vocabulary, the implementation selects the highest-logit token **only from the allowed tokens**.

Conceptually:

```text
LLM logits
    |
    v
All vocabulary tokens
    |
    v
Filter using current state
    |
    v
Allowed tokens only
    |
    v
Highest scoring allowed token
```

The subject describes the same principle: invalid tokens are excluded so that the generated sequence remains compatible with the required structure and schema.

---

# State Transitions

After a token is generated, the current state decides whether generation should continue or move to another state.

For example:

```text
FunctionNameState
        |
        | complete function name
        v
ParametersState
```

Then:

```text
ParametersState
        |
        | "("
        v
ParameterNameState
```

After a parameter name:

```text
ParameterNameState
        |
        | ":"
        v
ParameterValueState
```

When a parameter value can finish:

```text
ParameterValueState
        |
        +---- "," ----> next parameter
        |
        +---- ")" ----> function call complete
```

The separator is only allowed when the current value is complete.

This prevents a comma or closing parenthesis from prematurely terminating an incomplete value.

---

# Why `can_finish()` Is Separate From Prefix Validation

A parameter value can be a valid prefix without being a complete value.

For example, for a number:

```text
-
```

is a possible prefix of:

```text
-42
```

and:

```text
12.
```

is a possible prefix of:

```text
12.5
```

Therefore the parameter rules distinguish between:

```python
_is_valid_prefix()
```

and:

```python
can_finish()
```

`_is_valid_prefix()` determines whether generation may continue.

`can_finish()` determines whether the decoder is allowed to terminate the parameter.

This keeps the logic for parameter validation inside the parameter-value rules instead of duplicating it inside the generation state machine.

---

# Vocabulary Cache

The implementation uses a token-text cache:

```python
_token_text_cache
```

to avoid repeatedly decoding the same token ID.

This is particularly useful because constrained decoding checks many vocabulary tokens at every generation step.

Instead of repeatedly performing:

```python
model.decode([token_id])
```

for the same token, the decoded text can be reused from the cache.

---

# Output

The final output is written to:

```text
data/output/function_calling_results.json
```

The output contains one object per input prompt.

Example:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {
            "s": "hello"
        }
    }
]
```

Each result contains exactly:

```text
prompt
name
parameters
```

The subject requires valid JSON, no additional prose, all required parameters, and parameter types matching the function definition.

---

# Design Decisions

## State Machine

A state-machine design was chosen because the output has a known structure.

Each state has one responsibility:

* Select a function.
* Generate the opening parenthesis.
* Generate a parameter name.
* Generate the separator.
* Generate the parameter value.
* Move to the next parameter or finish.

This makes the constrained-decoding logic easier to understand and extend.

## Parameter-Specific Rules

Parameter validation is separated by type.

Instead of having one large validation function containing all possible types, each rule is responsible for its own constraints.

This keeps number, integer, boolean, and string behaviour independent.

## Shared Generation Context

`GenerationContext` stores information shared between states.

This avoids passing the same collection of values from one state to another manually.

## LLM-Based Function Selection

The function is selected by the LLM from the available function descriptions.

The implementation does not use a keyword-based function-selection system.

Constrained decoding is then used to ensure that the generated function name corresponds to one of the available functions.

This follows the subject requirement that the function to call must be chosen using the LLM.

## Independent Prompts

Each user prompt is processed independently.

For every prompt, the program builds the instruction/function-definition prompt, tokenizes it, creates a fresh `GenerationContext`, and starts from `FunctionNameState`.

This prevents information from one test case from affecting another.

---

# Error Handling

The implementation checks for invalid situations during execution.

Examples include:

* Missing input files.
* Invalid JSON input.
* Unsupported parameter types.
* No allowed tokens for the current state.
* Generation exceeding the maximum token limit.
* Invalid generated parameter values.
* Unknown functions.

The generation loop also has a maximum token limit:

```text
200 tokens
```

If generation cannot complete within the limit, an error is raised with information about the current state and partial output.

The subject requires graceful error handling and clear error messages rather than unexpected crashes.

---

# Performance

The subject targets:

* 90%+ function-selection and argument-extraction accuracy.
* 100% valid JSON.
* Processing the test prompts in under 5 minutes on standard hardware.
* Robust handling of errors and edge cases.

The implementation measures execution time for individual prompts and for the complete test run.

The main performance cost comes from constrained autoregressive generation because the model must calculate logits repeatedly while generating the function call token by token.

The implementation also reduces repeated tokenizer work through token caching.

### Performance results

Results should be recorded using the actual machine and test set used during evaluation:

```text
Model:
Number of prompts:
Total execution time:
Average time per prompt:
Correct function selections:
Correct argument extractions:
Valid JSON outputs:
```

Accuracy should be measured against the expected function calls rather than assuming that a syntactically valid JSON response is automatically semantically correct.

---

# Testing Strategy

The implementation should be tested using the provided test file as well as additional edge cases.

## Basic function selection

Examples:

```text
What is the sum of 2 and 3?
Greet John
Reverse the string 'hello'
```

## Numbers

Test:

```text
0
42
265
-10
-42.75
12.0
```

## Strings

Test:

```text
hello
"hello world"
file paths
strings containing spaces
special characters
```

## Boolean values

Test:

```text
true
false
yes
no
1
0
```

## Multiple parameters

Functions with multiple parameters should be tested to verify that:

* Parameter names are generated in the correct order.
* Commas are generated correctly.
* Each parameter uses the correct validation rule.
* The final parameter is followed by `)`.

## Error cases

The implementation should also be tested with:

* Missing input files.
* Invalid JSON.
* Empty prompts.
* Ambiguous prompts.
* Large numbers.
* Negative numbers.
* Unexpected parameter types.
* Special characters.
* Functions with multiple parameters.

These cases correspond to the testing guidance in the subject.

---

# Example

Input:

```json
[
    {
        "prompt": "Reverse the string 'hello'"
    }
]
```

The model receives the function definitions and the user request.

The generation process then follows approximately:

```text
FunctionNameState
        |
        v
fn_reverse_string
        |
        v
ParametersState
        |
        v
(
        |
        v
ParameterNameState
        |
        v
s
        |
        v
ParameterValueSeparatorState
        |
        v
:
        |
        v
ParameterValueState
        |
        v
hello
        |
        v
)
```

Final result:

```json
[
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {
            "s": "hello"
        }
    }
]
```

---

# Running With Make

Run the default project:

```bash
make run
```

Install dependencies:

```bash
make install
```

Run linting:

```bash
make lint
```

Run the debugger:

```bash
make debug
```

Clean generated Python caches:

```bash
make clean
```

---

# Limitations

The current implementation focuses on the mandatory function-calling task.

The generation process is intentionally token-level and therefore can be slower than unconstrained generation because the allowed token set is evaluated repeatedly.

The current implementation also relies on the interfaces provided by `Small_LLM_Model`, including:

```python
encode()
decode()
get_logits_from_input_ids()
get_path_to_vocab_file()
```

Model compatibility therefore depends on the tokenizer and vocabulary behaviour exposed through the SDK.

---

# Possible Improvements

Possible future improvements include:

* More efficient vocabulary/token caching.
* Further reduction of repeated tokenization.
* More comprehensive automated tests.
* Additional supported parameter types.
* More advanced nested object/array handling.
* More efficient constrained-token lookup.
* Additional compatible LLM models.
* Visualization of token-by-token generation.

These improvements correspond to several of the optional bonus directions described by the subject, including multiple models, caching/performance optimizations, comprehensive testing, visualization, and nested arguments.

---

# Resources

## 42 Subject

The primary reference for this project is the official **Call Me Maybe** project subject.

The subject covers:

* Function calling.
* LLM interaction.
* Tokenization.
* Logits.
* Constrained decoding.
* JSON schema validation.
* Performance and reliability.
* Testing requirements.

## LLM Concepts

Useful topics for understanding this project:

* Large Language Models
* Tokenization
* Tokens and vocabulary
* Logits
* Autoregressive generation
* Constrained decoding
* Structured output
* JSON schema validation

The provided project subject explains the generation pipeline as:

```text
Prompt
    ->
Tokenization
    ->
Input IDs
    ->
LLM
    ->
Logits
    ->
Next Token Selection
```

and repeats this process until generation is complete.

---

# AI Usage

AI tools were used as a learning and development aid during this project.

AI assistance was used for:

* Understanding LLM tokenization and logits.
* Understanding constrained decoding.
* Discussing state-machine design.
* Debugging Python code.
* Improving error handling.
* Reviewing regular expressions and parameter validation.
* Understanding model/tokenizer behaviour.
* Reviewing code structure and simplifying implementation.
* Writing and improving documentation.

The final implementation was reviewed, tested, and adapted by the project author.

AI was not used as a replacement for understanding the project requirements or the constrained-decoding algorithm.

---

# Requirements Checklist

| Requirement                 | Implementation |
| --------------------------- | -------------- |
| Python 3.10+                | Yes            |
| Function selected using LLM | Yes            |
| Constrained decoding        | Yes            |
| Function-name constraint    | Yes            |
| Parameter-name constraint   | Yes            |
| Number validation           | Yes            |
| Integer validation          | Yes            |
| Boolean validation          | Yes            |
| String validation           | Yes            |
| Typed JSON output           | Yes            |
| Multiple parameters         | Yes            |
| Error handling              | Yes            |
| `uv` dependency management  | Yes            |
| Makefile                    | Yes            |
| Type hints                  | Yes            |
| PEP 257 docstrings          | Yes            |
| Pydantic classes            | Yes            |
| Configurable model          | Yes            |
| README documentation        | Yes            |

---

# Conclusion

Call Me Maybe demonstrates how a relatively small language model can be guided to produce structured function calls using constrained decoding.

Instead of trusting the LLM to generate valid structured output on its own, the implementation controls the generation process token by token.

The LLM provides the semantic decision-making required to select the appropriate function, while the state machine and parameter rules constrain the generated output to the required structure and parameter types.

This separation between **LLM reasoning** and **structural constraints** is the central idea of the project.
