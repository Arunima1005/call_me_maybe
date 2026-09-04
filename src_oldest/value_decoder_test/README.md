
# Decoder test

This is a small isolated test of the value-decoder concept.

Run:

    python test_decoder.py

If pytest is installed:

    pytest -q

The tests do not download a Hugging Face model. They use a FakeModel
whose decode() simply joins vocabulary strings.

After these tests pass, connect ValueDecoderFactory to ExpectValue and
then to the real LLM logits.
