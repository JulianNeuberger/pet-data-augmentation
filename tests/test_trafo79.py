from augment.trafo79 import Trafo79Step
from data import model


def test_do_augment():
    tokens = [
        model.Token(
            text="I",
            index_in_document=0,
            pos_tag="PRP",
            bio_tag="B-Actor",
            sentence_index=0,
        ),
        model.Token(
            text="leave",
            index_in_document=1,
            pos_tag="VBP",
            bio_tag="O",
            sentence_index=0,
        ),
        model.Token(
            text="Human",
            index_in_document=2,
            pos_tag="NN",
            bio_tag="B-Activity Data",
            sentence_index=0,
        ),
        model.Token(
            text="resources",
            index_in_document=3,
            pos_tag="NNS",
            bio_tag="I-Activity Data",
            sentence_index=0,
        ),
        model.Token(
            text="easy",
            index_in_document=4,
            pos_tag="JJ",
            bio_tag="O",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=5, pos_tag=".", bio_tag=".", sentence_index=0
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    doc = model.Document(
        text="I leave Human resources easy.",
        name="1",
        sentences=[sentence],
        mentions=[model.Mention("Actor", 0, [0]), model.Mention("Object", 0, [2, 3])],
        entities=[],
        relations=[],
    )

    trafo = Trafo79Step([doc], p=1)
    augmented = trafo.do_augment(doc)
    assert len(augmented.tokens) == 0

    trafo = Trafo79Step([doc], p=0)
    augmented = trafo.do_augment(doc)
    assert len(augmented.tokens) == len(doc.tokens)
