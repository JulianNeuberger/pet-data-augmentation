import copy

from augment import trafo101
from data import model


def test_do_augment():
    # Arrange

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
            text="worker",
            index_in_document=3,
            pos_tag="NN",
            bio_tag="",
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
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=8, pos_tag=".", bio_tag="", sentence_index=0
        ),
    ]

    tokens2 = copy.deepcopy(tokens)
    tokens2.pop()
    tokens2.append(
        model.Token(
            text="available",
            index_in_document=8,
            pos_tag="JJ",
            bio_tag="",
            sentence_index=0,
        )
    )
    tokens2.append(
        model.Token(
            text=".", index_in_document=9, pos_tag=".", bio_tag="", sentence_index=0
        )
    )

    doc = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department available.",
        name="1",
        sentences=[model.Sentence(tokens), model.Sentence(tokens2)],
        mentions=[
            model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
            model.Mention(
                ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5]
            ),
        ],
        entities=[],
        relations=[],
    )

    trafo = trafo101.Trafo101Step([doc], n=1)
    aug = trafo.do_augment(doc)

    print()
    print(" ".join(t.text for t in doc.tokens))
    print(" ".join(t.text for t in aug.tokens))
