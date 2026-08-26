from src.value_decoder_test.expect import ExpectValue


class FakeModel:
    def __init__(self, vocabulary, logits):
        self.vocabulary = vocabulary
        self.logits = logits

    def decode(self, token_ids):
        return "".join(
            self.vocabulary[token_id]
            for token_id in token_ids
        )

    def get_logits_from_input_ids(self, input_ids):
        return self.logits.copy()


class FakeContext:
    def __init__(self):
        self.functions = {}
        self.parameters = []
        self.param_def = ()
        self.kvsep = ()
        self.sep = ()
        self.end = ()


VOCAB = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "-",
    ".",
    "abc",
]


def select_best_allowed_token(
    model,
    rule,
):
    allowed = rule.get_allowed()

    logits = model.get_logits_from_input_ids([])

    valid_logits = {
        token_id: logits[token_id]
        for token_id in allowed
    }

    return max(
        valid_logits,
        key=valid_logits.get,
    )


def test_invalid_highest_logit_is_rejected():

    abc_id = VOCAB.index("abc")
    twelve_id = VOCAB.index("1")

    logits = [
        0.0
        for _ in VOCAB
    ]

    # Deliberately make the INVALID token
    # the highest-logit token.
    logits[abc_id] = 0.99

    # Valid token.
    logits[twelve_id] = 0.80

    model = FakeModel(
        VOCAB,
        logits,
    )

    rule = ExpectValue(
        context=FakeContext(),
        model=model,
        vocabulary=VOCAB,
        parameter_types={},
        parameter_type="number",
    )

    selected_token = (
        select_best_allowed_token(
            model,
            rule,
        )
    )

    assert selected_token != abc_id
    assert selected_token == twelve_id


if __name__ == "__main__":
    test_invalid_highest_logit_is_rejected()

    print(
        "GENERATION CONSTRAINT TEST PASSED"
    )