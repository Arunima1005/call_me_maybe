import json

from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    print("Vocabulary type:", type(vocab))
    print("Number of entries:", len(vocab))

    id_to_token = {
    token_id: token
    for token, token_id in vocab.items()
}
    # print(repr(id_to_token[3555]))
    # print(model.decode([3555]))

    text = '{"name": "fn_add_numbers"}'

    token_ids = model.encode(text)

    print("Token IDs:")
    print(token_ids)

    print("\nTokens:")

    for token_id in token_ids[0].tolist():
        print(
            token_id,
            repr(id_to_token[token_id]),
            "->",
            repr(model.decode([token_id])),
        )

    # print("\nFirst 10 vocabulary entries:")

    # for i, (token, token_id) in enumerate(vocab.items()):
    #     print(repr(token), "->", token_id)

    #     if i == 9:
    #         break

    # for token, token_id in vocab.items():
    #     if token_id == 3555:
    #         print("Found:")
    #         print("Raw vocabulary token:", repr(token))
    #         print("Token ID:", token_id)

    # token = " What"

    # print("\nToken lookup:")
    # print(repr(token), "->", vocab.get(token))
    


if __name__ == "__main__":
    main()