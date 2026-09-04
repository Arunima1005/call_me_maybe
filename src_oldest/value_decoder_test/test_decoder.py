
from decoder import (
    BooleanDecoder,
    IntegerDecoder,
    NumberDecoder,
    StringDecoder,
)


class FakeModel:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def decode(self, token_ids):
        return "".join(self.vocabulary[i] for i in token_ids)


VOCAB = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "-", ".", "t", "r", "u", "e", "f", "a", "l", "s",
    '"', ",", "}", "EUR", "abc", "5,", "12.5", '"EUR"',
]


def allowed_texts(decoder):
    return {
        decoder.vocabulary[i]
        for i in decoder.get_allowed_tokens()
    }


def test_integer():
    model = FakeModel(VOCAB)
    d = IntegerDecoder(model, VOCAB)

    allowed = allowed_texts(d)

    assert "1" in allowed
    assert "-" in allowed
    assert "abc" not in allowed
    assert "." not in allowed

    d.add_token(VOCAB.index("1"))
    d.add_token(VOCAB.index("2"))

    assert d.get_value() == "12"
    assert d.can_finish()


def test_number():
    model = FakeModel(VOCAB)
    d = NumberDecoder(model, VOCAB)

    d.add_token(VOCAB.index("1"))
    d.add_token(VOCAB.index("2"))
    d.add_token(VOCAB.index("."))

    assert d.get_value() == "12."
    assert not d.can_finish()

    # This token contains the rest of the value.
    d2 = NumberDecoder(model, VOCAB)
    d2.add_token(VOCAB.index("12.5"))
    assert d2.get_value() == "12.5"
    assert d2.can_finish()


def test_boolean():
    model = FakeModel(VOCAB)
    d = BooleanDecoder(model, VOCAB)

    d.add_token(VOCAB.index("t"))
    assert not d.can_finish()

    d.add_token(VOCAB.index("r"))
    d.add_token(VOCAB.index("u"))
    d.add_token(VOCAB.index("e"))

    assert d.get_value() == "true"
    assert d.can_finish()


def test_string():
    model = FakeModel(VOCAB)
    d = StringDecoder(model, VOCAB)

    d.add_token(VOCAB.index('"EUR"'))

    assert d.get_value() == '"EUR"'
    assert d.can_finish()


def test_delimiter_in_same_token():
    model = FakeModel(VOCAB)
    d = IntegerDecoder(model, VOCAB)

    d.add_token(VOCAB.index("5,"))

    result = d.analyse(d.get_value())

    assert result.valid
    assert result.complete
    assert result.value == "5"
    assert result.remainder == ","


if __name__ == "__main__":
    test_integer()
    test_number()
    test_boolean()
    test_string()
    test_delimiter_in_same_token()
    print("ALL TESTS PASSED")
