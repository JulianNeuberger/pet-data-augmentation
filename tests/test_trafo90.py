from augment.trafo90 import Trafo90Step
from data import model


def test_do_augment():
    # test1
    tokens = []
    tokens.append(
        model.Token(
            text="I",
            index_in_document=0,
            pos_tag="PRP",
            bio_tag="B-Actor",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text="am",
            index_in_document=1,
            pos_tag="VBP",
            bio_tag="I-Actor",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text="the", index_in_document=2, pos_tag="DT", bio_tag="O", sentence_index=0
        )
    )
    tokens.append(
        model.Token(
            text="Head",
            index_in_document=3,
            pos_tag="NN",
            bio_tag="B-Activity",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text="of",
            index_in_document=4,
            pos_tag="IN",
            bio_tag="I-Activity",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text="the", index_in_document=5, pos_tag="DT", bio_tag="O", sentence_index=0
        )
    )
    tokens.append(
        model.Token(
            text="functional",
            index_in_document=6,
            pos_tag="JJ",
            bio_tag="O",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text="department",
            index_in_document=7,
            pos_tag="NN",
            bio_tag="O",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text=".", index_in_document=8, pos_tag=".", bio_tag="O", sentence_index=0
        )
    )

    sentence1 = model.Sentence(tokens)
    sentence2 = sentence1.copy()
    sentence2.tokens[5].bio_tag = "B-Actor"
    sentence2.tokens[6].bio_tag = "I-Actor"
    sentence2.tokens[7].bio_tag = "B-Activity"

    for i in range(9):
        sentence2.tokens[i].sentence_index = 1

    doc = model.Document(
        text="I am the Head of the functional department.",
        name="1",
        sentences=[sentence1, sentence2],
        mentions=[
            model.Mention("Actor", 0, [0]),
            model.Mention("Actor", 1, [5, 6]),
            model.Mention("Activity", 1, [7]),
        ],
        entities=[],
        relations=[],
    )

    trafo1 = Trafo90Step([doc], prob=1)
    aug = trafo1.do_augment(doc)

    print(" ".join(t.text for t in aug.tokens))
