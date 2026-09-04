from value_decoder_test.parameter_values_old import (
    BooleanParameter,
    IntegerParameter,
    NumberParameter,
    StringParameter,
)


class FakeModel:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def decode(self, token_ids):
        return "".join(
            self.vocabulary[i]
            for i in token_ids
        )


VOCAB = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
    "-", ".", "t", "r", "u",
    "e", "f", "a", "l", "s",
    '"', ",", "}", "EUR", "abc",
    "5,", "12.5", '"EUR"',
]


def allowed_texts(rule):
    return {
        rule.vocabulary[i]
        for i in rule.get_allowed_tokens()
    }


def test_integer():
    model = FakeModel(VOCAB)
    rule = IntegerParameter(model, VOCAB)

    allowed = allowed_texts(rule)

    assert "1" in allowed
    assert "-" in allowed
    assert "abc" not in allowed
    assert "." not in allowed

    rule.add_token(VOCAB.index("1"))
    rule.add_token(VOCAB.index("2"))

    assert rule.get_value() == "12"
    assert rule.can_finish()


def test_number():
    model = FakeModel(VOCAB)
    rule = NumberParameter(model, VOCAB)

    rule.add_token(VOCAB.index("1"))
    rule.add_token(VOCAB.index("2"))
    rule.add_token(VOCAB.index("."))

    assert rule.get_value() == "12."
    assert not rule.can_finish()

    other = NumberParameter(model, VOCAB)
    other.add_token(VOCAB.index("12.5"))

    assert other.get_value() == "12.5"
    assert other.can_finish()


def test_boolean():
    model = FakeModel(VOCAB)
    rule = BooleanParameter(model, VOCAB)

    rule.add_token(VOCAB.index("t"))

    assert not rule.can_finish()

    rule.add_token(VOCAB.index("r"))
    rule.add_token(VOCAB.index("u"))
    rule.add_token(VOCAB.index("e"))

    assert rule.get_value() == "true"
    assert rule.can_finish()


def test_string():
    model = FakeModel(VOCAB)
    rule = StringParameter(model, VOCAB)

    rule.add_token(VOCAB.index('"EUR"'))

    assert rule.get_value() == '"EUR"'
    assert rule.can_finish()


def test_value_and_delimiter_in_one_token():
    model = FakeModel(VOCAB)
    rule = IntegerParameter(model, VOCAB)

    rule.add_token(VOCAB.index("5,"))

    result = rule.inspect(rule.get_value())

    assert result.accepted
    assert result.finished
    assert result.value == "5"
    assert result.tail == ","


def test_number_prefix_minus():
    model = FakeModel(VOCAB)
    rule = NumberParameter(model, VOCAB)

    rule.add_token(VOCAB.index("-"))

    assert rule.get_value() == "-"
    assert not rule.can_finish()


if __name__ == "__main__":
    test_integer()
    test_number()
    test_boolean()
    test_string()
    test_value_and_delimiter_in_one_token()
    test_number_prefix_minus()

    print("ALL TESTS PASSED")