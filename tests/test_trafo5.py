import copy
from data import model
from augment.trafo5 import Trafo5Step


# Author for entire script: Benedikt
def test_is_ant():
    trafo = Trafo5Step([])

    # Arrange
    word1 = "evil"
    word2 = "good"
    word3 = "good"
    word4 = "nice"
    sol1 = True
    sol2 = False

    # Act
    is_ant1 = trafo.are_antonym(word1, word2)
    is_ant2 = trafo.are_antonym(word3, word4)
    # Assert
    assert is_ant1 == sol1
    assert is_ant2 == sol2


def test_is_syn():
    # Arrange
    trafo = Trafo5Step([])
    word1 = "evil"
    word2 = "good"
    word3 = "people"
    word4 = "masses"
    sol1 = False
    sol2 = True

    # Act
    is_ant1 = trafo.are_synonym(word1, word2)
    is_ant2 = trafo.are_synonym(word3, word4)

    # Assert
    assert is_ant1 == sol1
    assert is_ant2 == sol2


def test_do_augment():
    # Arrange
    trafo = Trafo5Step([])
    tokens = []
    tokens.append(
        model.Token(
            text="I", index_in_document=0, pos_tag="PRP", bio_tag="", sentence_index=0
        )
    )
    tokens.append(
        model.Token(
            text="am", index_in_document=1, pos_tag="VBP", bio_tag="", sentence_index=0
        )
    )
    tokens.append(
        model.Token(
            text="good", index_in_document=2, pos_tag="JJ", bio_tag="", sentence_index=0
        )
    )
    tokens.append(
        model.Token(
            text="worker",
            index_in_document=3,
            pos_tag="NN",
            bio_tag="",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text="of", index_in_document=4, pos_tag="IN", bio_tag="", sentence_index=0
        )
    )
    tokens.append(
        model.Token(
            text="the", index_in_document=5, pos_tag="DT", bio_tag="", sentence_index=0
        )
    )
    tokens.append(
        model.Token(
            text="functional",
            index_in_document=6,
            pos_tag="JJ",
            bio_tag="",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text="department",
            index_in_document=7,
            pos_tag="NN",
            bio_tag="",
            sentence_index=0,
        )
    )
    tokens.append(
        model.Token(
            text=".", index_in_document=8, pos_tag=".", bio_tag="", sentence_index=0
        )
    )
    sentence1 = model.Sentence(tokens=tokens)

    doc = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department available.",
        name="1",
        sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[],
    )

    doc2 = copy.deepcopy(doc)
    doc2.sentences[0].tokens.pop(2)

    doc3 = copy.deepcopy(doc)
    doc3.sentences[0].tokens[6].text = "evil"

    doc_sol1 = copy.deepcopy(doc)
    doc_sol1.sentences[0].tokens[2].text = "bad"
    doc_sol1.sentences[0].tokens[6].text = "nonfunctional"

    doc_sol2 = copy.deepcopy(doc2)

    doc_sol3 = copy.deepcopy(doc3)
    # Act

    doc_aug1 = trafo.do_augment(doc)
    doc_aug2 = trafo.do_augment(doc2)
    doc_aug3 = trafo.do_augment(doc3)
    # Assert

    assert doc_aug1 == doc_sol1
    assert doc_aug2 == doc_sol2
    assert doc_aug3 == doc_sol3
