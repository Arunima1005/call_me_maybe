from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    for text in [" Paris", " London", " Rome"]:
        token_ids = model.encode(text)[0].tolist()

        print(repr(text), "->", token_ids)

        for token_id in token_ids:
            print(
                "   ",
                token_id,
                "->",
                repr(model.decode([token_id])),
            )


if __name__ == "__main__":
    main()