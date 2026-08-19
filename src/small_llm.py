from llm_sdk.llm_sdk import Small_LLM_Model

def main() -> None:
    model = Small_LLM_Model()

    text = "What is the sum of 2 and 3?"

    token_ids = model.encode(text)

    print("Token IDs:")
    print(token_ids)

    input_ids = token_ids[0].tolist()

    logits = model.get_logits_from_input_ids(input_ids)

    next_token_id = logits.index(max(logits))

    print("\nNext token ID:")
    print(next_token_id)

    print("\nNext token:")
    print(model.decode([next_token_id]))

    vocab_path = model.get_path_to_vocab_file()

    print("\nVocabulary path:")
    print(vocab_path)


if __name__ == "__main__":
    main()