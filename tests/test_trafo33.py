import copy

import mockito

from augment import trafo33
from data import model

# Author for entire script: Benedikt
def test_do_augment():
    # Arrange
    trafo1 = trafo33.Trafo33Step()
    # Testfall1
    tokens = [model.Token(text="because", index_in_document=0,
                          pos_tag="JJ", bio_tag="O",
                          sentence_index=0),
              model.Token(text="the", index_in_document=1,
                          pos_tag="VBP", bio_tag="O",
                          sentence_index=0),
              model.Token(text="contrary", index_in_document=2,
                          pos_tag="NN", bio_tag="O",
                          sentence_index=0),
              model.Token(text="head", index_in_document=3,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0),
              model.Token(text=".", index_in_document=4,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0)
              ]
    sentence1 = model.Sentence(tokens=tokens)

    doc = model.Document(
        text="good leave head .",
        name="1", sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[])

    tokens_sol1 = [model.Token(text="inasmuch", index_in_document=0,
                          pos_tag="JJ", bio_tag="O",
                          sentence_index=0),
              model.Token(text="as", index_in_document=1,
                          pos_tag="IN", bio_tag="O",
                          sentence_index=0),
              model.Token(text="the", index_in_document=2,
                          pos_tag="VBP", bio_tag="O",
                          sentence_index=0),
              model.Token(text="contrary", index_in_document=3,
                          pos_tag="NN", bio_tag="O",
                          sentence_index=0),
              model.Token(text="head", index_in_document=4,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0),
              model.Token(text=".", index_in_document=5,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0)
              ]
    sentence1_sol1 = model.Sentence(tokens=tokens_sol1)
    doc_sol1 = model.Document(
        text="good leave head .",
        name="1", sentences=[sentence1_sol1],
        mentions=[],
        entities=[],
        relations=[])
    doc_to_aug1 = copy.deepcopy(doc)



    #Testfall 2
    tokens2 = [model.Token(text="for", index_in_document=0,
                          pos_tag="JJ", bio_tag="O",
                          sentence_index=0),
              model.Token(text="example", index_in_document=1,
                          pos_tag="VBP", bio_tag="O",
                          sentence_index=0),
              model.Token(text="contrary", index_in_document=2,
                          pos_tag="NN", bio_tag="O",
                          sentence_index=0),
              model.Token(text="head", index_in_document=3,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0),
              model.Token(text=".", index_in_document=4,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0)
              ]
    sentence2 = model.Sentence(tokens=tokens2)
    doc2 = model.Document(
        text="good leave head .",
        name="1", sentences=[sentence2],
        mentions=[],
        entities=[],
        relations=[])

    tokens_sol2 = [model.Token(text="for", index_in_document=0,
                               pos_tag="IN", bio_tag="O",
                               sentence_index=0),
                   model.Token(text="instance", index_in_document=1,
                               pos_tag="NN", bio_tag="O",
                               sentence_index=0),
                   model.Token(text="contrary", index_in_document=2,
                               pos_tag="NN", bio_tag="O",
                               sentence_index=0),
                   model.Token(text="head", index_in_document=3,
                               pos_tag=".", bio_tag="O",
                               sentence_index=0),
                   model.Token(text=".", index_in_document=4,
                               pos_tag=".", bio_tag="O",
                               sentence_index=0)
                   ]
    sentence1_sol2 = model.Sentence(tokens=tokens_sol2)
    doc_sol2 = model.Document(
        text="good leave head .",
        name="1", sentences=[sentence1_sol2],
        mentions=[],
        entities=[],
        relations=[])
    doc_to_aug2 = copy.deepcopy(doc2)
    # Act
    mockito.when(trafo33).randint(0, 3).thenReturn(1)
    doc_aug1 = trafo1.do_augment(doc_to_aug1)
    mockito.when(trafo33).randint(0, 2).thenReturn(1)
    doc_aug2 = trafo1.do_augment(doc_to_aug2)

    # Assert
    assert doc_aug1 == doc_sol1
    assert doc_aug2 == doc_sol2