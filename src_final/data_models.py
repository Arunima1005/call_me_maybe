from pydantic import BaseModel

class Function_definition(BaseModel):
    name: str
    description: str
    parameters : dict[str, dict[str, str]]
    returns : dict[str, str]

class GenerationContext(BaseModel):
    functions: list[Function_definition]
    function_token_sequences: list[list[int]]

    model: object
    vocabulary: list[str]
    open_parameter_token: int
    key_value_separator_token: int
    parameter_separator_token: int
    close_parameter_token: int

    # current_parameter: dict[str, str] | None = None
    selected_function: Function_definition | None = None
    current_parameter_index: int = 0
    parameter_values: dict[str, object] = {}

    literal_candidates: list[str] = [] 
    user_prompt: str = "" 
