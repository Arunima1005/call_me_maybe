from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    text = "What is the sum of 2 and 3?"

    token_ids = model.encode(text)

    print("Token IDs:")
    print(token_ids)

    input_ids = token_ids[0].tolist()
    print(len(input_ids))

    logits = model.get_logits_from_input_ids(input_ids)

    print("\nNumber of logits:")
    print(len(logits))

    next_token_id = max(
        range(len(logits)),
        key=lambda i: logits[i],
    )

    print("\nNext token ID:")
    print(next_token_id)

    print("\nNext token:")
    print(model.decode([next_token_id]))


if __name__ == "__main__":
    main()