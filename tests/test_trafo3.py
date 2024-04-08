import copy

from augment.trafo3 import Trafo3Step
from data import model


def test_do_augment():
    # test1 - no adjectives in the sentence
    tokens = [model.Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="", sentence_index=0),
              model.Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="", sentence_index=0),
              model.Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="", sentence_index=0),
              model.Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="", sentence_index=0),
              model.Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="", sentence_index=0),
              model.Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="", sentence_index=0),
              model.Token(text="department", index_in_document=6, pos_tag="NN", bio_tag="", sentence_index=0),
              model.Token(text=".", index_in_document=7, pos_tag=".", bio_tag="", sentence_index=0)]

    sentence = model.Sentence(tokens=tokens)
    mentions1 = [model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
                 model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5])]

    doc1 = model.Document(text="I am the Head of the department.", name="1", sentences=[sentence, sentence],
                         mentions=mentions1, entities=[], relations=[])
    doc_sol1 = copy.deepcopy(doc1)

    # test2 - (Parameters: no_dupl = False) duplicates allowed
    tokens2 = [model.Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="", sentence_index=0),
               model.Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="", sentence_index=0),
               model.Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="", sentence_index=0),
               model.Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="", sentence_index=0),
               model.Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="", sentence_index=0),
               model.Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="", sentence_index=0),
               model.Token(text="functional", index_in_document=6, pos_tag="JJ", bio_tag="", sentence_index=0),
               model.Token(text="department", index_in_document=7, pos_tag="NN", bio_tag="",  sentence_index=0),
               model.Token(text=".", index_in_document=8, pos_tag=".", bio_tag="", sentence_index=0)]

    sentence1 = model.Sentence(tokens=tokens2)

    mentions2 = [model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
                 model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5])]

    doc2 = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department available.",
        name="1", sentences=[sentence1], mentions=mentions2, entities=[], relations=[])

    doc_sol2 = copy.deepcopy(doc2)
    doc_sol2.sentences[0].tokens[6].text = "nonfunctional"

    # ACT
    # test1
    doc_to_aug1 = copy.deepcopy(doc1)
    trafo1 = Trafo3Step([doc_to_aug1], 1)
    doc_aug1 = trafo1.do_augment(doc_to_aug1)

    # test2
    doc_to_aug2 = copy.deepcopy(doc2)
    doc_aug2 = trafo1.do_augment(doc_to_aug2)

    # Assert
    assert doc_aug1 == doc_sol1
    assert " ".join(t.text for t in doc_aug2.tokens) == " ".join(t.text for t in doc_sol2.tokens)
