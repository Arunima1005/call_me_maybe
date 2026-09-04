from value_decoder_test.parameter_values_old import (
    IntegerParameter,
    NumberParameter,
    BooleanParameter,
    StringParameter,
)

from src.value_decoder_test.expect import ExpectValue

class FakeModel:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def decode(self, token_ids):
        return "".join(
            self.vocabulary[token_id]
            for token_id in token_ids
        )


class FakeContext:
    """
    Minimal context needed by ExpectValue.
    """

    def __init__(self):
        self.functions = {}
        self.parameters = []
        self.param_def = ()
        self.kvsep = ()
        self.sep = ()
        self.end = ()


VOCAB = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
    "-",
    ".",
    "t", "r", "u", "e",
    "f", "a", "l", "s",
    '"',
    ",",
    "}",
    "EUR",
    "abc",
    "12.5",
    '"EUR"',
]


def test_expect_integer():

    model = FakeModel(VOCAB)
    context = FakeContext()

    rule = ExpectValue(
        context=context,
        model=model,
        vocabulary=VOCAB,
        parameter_types={},
        parameter_type="integer",
    )

    allowed = rule.get_allowed()

    allowed_text = {
        VOCAB[token_id]
        for token_id in allowed
    }

    assert "1" in allowed_text
    assert "-" in allowed_text
    assert "abc" not in allowed_text
    assert "." not in allowed_text

    rule.add_token(
        VOCAB.index("1")
    )

    rule.add_token(
        VOCAB.index("2")
    )

    assert rule.get_value() == "12"
    assert rule.can_finish()


def test_expect_number():

    model = FakeModel(VOCAB)
    context = FakeContext()

    rule = ExpectValue(
        context=context,
        model=model,
        vocabulary=VOCAB,
        parameter_types={},
        parameter_type="number",
    )

    rule.add_token(
        VOCAB.index("1")
    )

    rule.add_token(
        VOCAB.index("2")
    )

    rule.add_token(
        VOCAB.index(".")
    )

    assert rule.get_value() == "12."
    assert not rule.can_finish()

    other = ExpectValue(
        context=context,
        model=model,
        vocabulary=VOCAB,
        parameter_types={},
        parameter_type="number",
    )

    other.add_token(
        VOCAB.index("12.5")
    )

    assert other.get_value() == "12.5"
    assert other.can_finish()


def test_expect_boolean():

    model = FakeModel(VOCAB)
    context = FakeContext()

    rule = ExpectValue(
        context=context,
        model=model,
        vocabulary=VOCAB,
        parameter_types={},
        parameter_type="boolean",
    )

    for text in ["t", "r", "u", "e"]:

        rule.add_token(
            VOCAB.index(text)
        )

    assert rule.get_value() == "true"
    assert rule.can_finish()


def test_expect_string():

    model = FakeModel(VOCAB)
    context = FakeContext()

    rule = ExpectValue(
        context=context,
        model=model,
        vocabulary=VOCAB,
        parameter_types={},
        parameter_type="string",
    )

    rule.add_token(
        VOCAB.index('"EUR"')
    )

    assert rule.get_value() == '"EUR"'
    assert rule.can_finish()


if __name__ == "__main__":

    test_expect_integer()
    test_expect_number()
    test_expect_boolean()
    test_expect_string()

    print("EXPECT VALUE TESTS PASSED")