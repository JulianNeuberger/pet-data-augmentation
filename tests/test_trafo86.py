from augment.trafo86 import Trafo86HypernymReplacement, Trafo86HyponymReplacement
from data import model


def test_hypernym_replacement():
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="PRP", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="VBP", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="DT", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="Head",
            index_in_document=3,
            pos_tag="NN",
            bio_tag="B-Actor",
            sentence_index=0,
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="IN", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="DT", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="functional",
            index_in_document=6,
            pos_tag="JJ",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text="department",
            index_in_document=7,
            pos_tag="NN",
            bio_tag="B-Actor",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=8, pos_tag=".", bio_tag="", sentence_index=0
        ),
    ]

    doc = model.Document(text="", name="", sentences=[model.Sentence(tokens)])

    trafo = Trafo86HypernymReplacement([doc], n=1)
    aug = trafo.do_augment(doc)

    print()
    print("---------------------------------")
    print(" ".join(t.text for t in doc.tokens))
    print(" ".join(t.text for t in aug.tokens))


def test_hyponym_replacement():
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="PRP", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="VBP", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="DT", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="Head",
            index_in_document=3,
            pos_tag="NN",
            bio_tag="B-Actor",
            sentence_index=0,
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="IN", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="DT", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="functional",
            index_in_document=6,
            pos_tag="JJ",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text="department",
            index_in_document=7,
            pos_tag="NN",
            bio_tag="B-Actor",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=8, pos_tag=".", bio_tag="", sentence_index=0
        ),
    ]

    doc = model.Document(text="", name="", sentences=[model.Sentence(tokens)])

    trafo = Trafo86HyponymReplacement([doc], n=1)
    aug = trafo.do_augment(doc)

    print()
    print("---------------------------------")
    print(" ".join(t.text for t in doc.tokens))
    print(" ".join(t.text for t in aug.tokens))
