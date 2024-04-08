import copy
from augment.trafo40 import Trafo40Step
from augment import trafo40
import mockito
from data import model

# Author for entire script: Leonie
def test_do_augment():
    # ARRANGE
    # Trafo Objects for testing do_augment()
    trafo1 = Trafo40Step()
    trafo2 = Trafo40Step(insert_speaker_phrases= False, insert_filler_phrases= False, tags=["B-Actor", "B-Activity"])

    # test1 (Parameters: speaker_p = True, uncertain_p = True, filler_p = True, tags = None)
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="B-Actor", sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="I-Actor", sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="O", sentence_index=0))
    tokens.append(model.Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="B-Actor", sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="I-Actor", sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="O", sentence_index=0))
    tokens.append(model.Token(text="functional", index_in_document=6, pos_tag="JJ", bio_tag="B-Activity", sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=7, pos_tag="NN", bio_tag="O", sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=8, pos_tag=".", bio_tag="O", sentence_index=0))

    sentence1 = model.Sentence(tokens)

    doc = model.Document(
        text="I am the Head of the functional department.", name="1", sentences=[sentence1], mentions=[],
        entities=[], relations=[])

    doc_to_aug1 = copy.deepcopy(doc)
    doc_sol1 = copy.deepcopy(doc)

    tok1 = model.Token(text="actually", index_in_document=1, pos_tag="RB", bio_tag="I-Actor", sentence_index=0)
    doc_sol1.sentences[0].tokens.insert(1, tok1)

    tok2 = model.Token(text="probably", index_in_document=8, pos_tag="RB", bio_tag="I-Activity", sentence_index=0)
    doc_sol1.sentences[0].tokens.insert(8, tok2)

    for i in range(2, 8):
        doc_sol1.sentences[0].tokens[i].index_in_document += 1
    for i in range(9, 11):
        doc_sol1.sentences[0].tokens[i].index_in_document += 2


    # test2 (Parameters: speaker_p = False, uncertain_p = True, filler_p = False, tags = ["B-Actor", "B-Activity"])
    doc_to_aug2 = copy.deepcopy(doc)
    doc_sol2 = copy.deepcopy(doc_sol1)
    doc_sol2.sentences[0].tokens[1].text = "perhaps"

    # ACT
    # test1
    mockito.when(trafo40).rand().thenReturn(0.1).thenReturn(0.5).thenReturn(0.5).thenReturn(0.5).thenReturn(0.5).thenReturn(0.5).thenReturn(0.1).thenReturn(0.5)
    all = ['I think', 'I believe', 'I mean', 'I guess', 'that is', 'I assume', 'I feel', 'In my opinion', 'I would say', 'maybe', 'perhaps', 'probably', 'possibly', 'most likely', 'uhm', 'umm', 'ahh', 'err', 'actually', 'obviously', 'naturally', 'like', 'you know']
    mockito.when(trafo40).choice(all).thenReturn("actually").thenReturn("probably")
    doc_aug1 = trafo1.do_augment(doc_to_aug1)

    # test2
    unc_phr = ["maybe", "perhaps", "probably", "possibly", "most likely",]
    mockito.when(trafo40).rand().thenReturn(0.1).thenReturn(0.5).thenReturn(0.1)
    mockito.when(trafo40).choice(unc_phr).thenReturn("perhaps").thenReturn("probably")
    doc_aug2 = trafo2.do_augment(doc_to_aug2)

    # ASSERT
    assert doc_aug1 == doc_sol1  # test1
    assert doc_aug2 == doc_sol2  # test2