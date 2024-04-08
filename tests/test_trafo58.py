import copy

from augment.trafo58 import Trafo58Step
from data import model


def test_do_augment():
    trafo1 = Trafo58Step([], 1, "zh")
    tokens1 = [
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

    sentence1 = model.Sentence(tokens=tokens1)
    doc1 = model.Document(
        text="I leave Human resources",
        name="1",
        sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[],
    )
    tokens1_sol = [
        model.Token(
            text="I",
            index_in_document=0,
            pos_tag="PRP",
            bio_tag="B-Actor",
            sentence_index=0,
        ),
        model.Token(
            text="holidays",
            index_in_document=1,
            pos_tag="NNS",
            bio_tag="O",
            sentence_index=0,
        ),
        model.Token(
            text="humanities",
            index_in_document=2,
            pos_tag="NNS",
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
            text="for",
            index_in_document=4,
            pos_tag="IN",
            bio_tag="I-Activity Data",
            sentence_index=0,
        ),
        model.Token(
            text="resource",
            index_in_document=5,
            pos_tag="NN",
            bio_tag="I-Activity Data",
            sentence_index=0,
        ),
        model.Token(
            text="resources",
            index_in_document=6,
            pos_tag="NNS",
            bio_tag="I-Activity Data",
            sentence_index=0,
        ),
        model.Token(
            text="simple",
            index_in_document=7,
            pos_tag="NN",
            bio_tag="O",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=8, pos_tag=".", bio_tag=".", sentence_index=0
        ),
    ]
    sentence1_sol = model.Sentence(tokens=tokens1_sol)
    doc1_sol = model.Document(
        text="I leave Human resources",
        name="1",
        sentences=[sentence1_sol],
        mentions=[],
        entities=[],
        relations=[],
    )
    doc_to_aug1 = copy.deepcopy(doc1)

    # Act
    doc_aug1 = trafo1.do_augment(doc_to_aug1)

    # Assert
    assert doc_aug1 == doc1_sol


def test_candidate_collection():
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

    trafo = Trafo58Step(
        [doc], ["de", "fr", "ru"], "strict", num_translations=1, n=1, device=-1
    )

    candidates = trafo.get_sequences(doc)
    assert len(candidates) == 4
    assert len(candidates[0]) == 1
    assert candidates[0][0].text == "I"
    assert len(candidates[2]) == 2
    assert " ".join(t.text for t in candidates[2]) == "Human resources"


def test_sentence_generation():
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
        mentions=[],
        entities=[],
        relations=[],
    )

    trafo = Trafo58Step(
        [doc], ["de", "fr", "ru"], "strict", num_translations=1, n=1, device=-1
    )

    augmented = trafo.do_augment(doc)

    print()
    print("----")
    print("Original : ", " ".join([t.text for t in doc.tokens]))
    print("Augmented: ", " ".join([t.text for t in augmented.tokens]))
