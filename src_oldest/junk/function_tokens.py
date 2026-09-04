from llm_sdk.llm_sdk import Small_LLM_Model


FUNCTION_NAMES = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
]


def main() -> None:
    model = Small_LLM_Model()

    for name in FUNCTION_NAMES:
        token_ids = model.encode(name)[0].tolist()

        print(f"\n{name}")
        print("Token IDs:", token_ids)

        for token_id in token_ids:
            print(
                "  ",
                token_id,
                "->",
                repr(model.decode([token_id])),
            )


if __name__ == "__main__":
    main()