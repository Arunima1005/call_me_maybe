from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    text = "The capital of"

    input_ids = model.encode(text)[0].tolist()

    logits = model.get_logits_from_input_ids(input_ids)

    allowed_token_ids = {
        12095,  # Paris
        7148,   # London
        21718,  # Rome
    }

    # Find the best token among the allowed tokens only.
    next_token_id = max(
        allowed_token_ids,
        key=lambda token_id: logits[token_id],
    )

    print("Chosen token ID:", next_token_id)
    print("Chosen token:", repr(model.decode([next_token_id])))
    print("Logit:", logits[next_token_id])


if __name__ == "__main__":
    main()