from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    text = "The capital of France is"

    input_ids = model.encode(text)[0].tolist()

    for _ in range(10):
        logits = model.get_logits_from_input_ids(input_ids)

        next_token_id = logits.index(max(logits))

        input_ids.append(next_token_id)

        print(
            "ID:",
            next_token_id,
            "TOKEN:",
            repr(model.decode([next_token_id])),
        )


if __name__ == "__main__":
    main()