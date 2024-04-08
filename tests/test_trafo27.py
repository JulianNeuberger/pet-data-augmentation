from augment import trafo27
from data import model


def test_load():
    abbreviations = trafo27.Trafo27Step._load()

    assert len(abbreviations) > 0


def test_expansion():
    tokens = [model.Token("didn't", 0, "", "", 0)]
    document = model.Document(
        text="didn't",
        name="",
        sentences=[model.Sentence(tokens)],
    )

    trafo = trafo27.Trafo27Step([document])
    augmented = trafo.do_augment(document)

    assert len(augmented.tokens) == 2
    assert " ".join(t.text for t in augmented.tokens) == "did not"
