import copy
from augment import trafo103
from augment.trafo103 import Trafo103Step
from data import model
from data.model import Token, Sentence, Document
import mockito

# Author for entire script: Leonie
def test_do_augment():
    # ARRANGE
    trafo1 = Trafo103Step(num_of_words=2, prob=1, pos_tags=None)
    trafo2 = Trafo103Step(num_of_words=3, prob=1, pos_tags=['NN', 'IN', 'DT', 'NN'])
    trafo3 = Trafo103Step(num_of_words=3, prob=1, pos_tags=['NN', 'IN'])

    tokens1 = [Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="O", sentence_index=0),
               Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="O", sentence_index=0),
               Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="O", sentence_index=0),
               Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="O", sentence_index=0),
               Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="O", sentence_index=0),
               Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="O", sentence_index=0),
               Token(text="functional", index_in_document=6, pos_tag="JJ", bio_tag="O", sentence_index=0),
               Token(text="department", index_in_document=7, pos_tag="NN", bio_tag="O", sentence_index=0),
               Token(text=".", index_in_document=8, pos_tag=".", bio_tag="O", sentence_index=0)]
    tokens2 = [Token(text='When', index_in_document=0, pos_tag='WRB', bio_tag='O', sentence_index=1),
               Token(text='I', index_in_document=1, pos_tag='PRP', bio_tag='B-Actor', sentence_index=1),
               Token(text='have', index_in_document=2, pos_tag='VBP', bio_tag='O', sentence_index=1),
               Token(text='detected', index_in_document=3, pos_tag='VBN', bio_tag='B-Activity', sentence_index=1),
               Token(text='a', index_in_document=4, pos_tag='DT', bio_tag='B-Activity Data', sentence_index=1),
               Token(text='number', index_in_document=5, pos_tag='NN', bio_tag='I-Activity Data', sentence_index=1),
               Token(text='of', index_in_document=6, pos_tag='IN', bio_tag='I-Activity Data', sentence_index=1),
               Token(text='personnel', index_in_document=7, pos_tag='NN', bio_tag='I-Activity Data', sentence_index=1),
               Token(text='requirements', index_in_document=8, pos_tag='NNS', bio_tag='I-Activity Data', sentence_index=1),
               Token(text=',', index_in_document=9, pos_tag=',', bio_tag='O', sentence_index=1),
               Token(text='I', index_in_document=10, pos_tag='PRP', bio_tag='B-Actor', sentence_index=1),
               Token(text='report', index_in_document=11, pos_tag='VBP', bio_tag='B-Activity', sentence_index=1),
               Token(text='the', index_in_document=12, pos_tag='DT', bio_tag='B-Activity Data', sentence_index=1),
               Token(text='vacancy', index_in_document=13, pos_tag='NN', bio_tag='I-Activity Data', sentence_index=1),
               Token(text='to', index_in_document=14, pos_tag='IN', bio_tag='O', sentence_index=1),
               Token(text='the', index_in_document=15, pos_tag='DT', bio_tag='B-Actor', sentence_index=1),
               Token(text='Personnel', index_in_document=16, pos_tag='NNP', bio_tag='I-Actor', sentence_index=1),
               Token(text='Department', index_in_document=17, pos_tag='NNP', bio_tag='I-Actor', sentence_index=1),
               Token(text='.', index_in_document=18, pos_tag='.', bio_tag='O', sentence_index=1)]

    sentence1 = Sentence(tokens1)
    sentence2 = Sentence(tokens2)
    doc = Document(
        text="I am the Head of the functional department. When I have detected a number of personnel requirements , I report the vacancy to the Personnel Department .",
        name="1", sentences=[sentence1, sentence2], mentions=[], entities=[], relations=[])

    # test1 - tests case1 (no tags given) (Parameters: num_of_words=2, kind_of_word= None)
    doc_to_aug1 = copy.deepcopy(doc)
    doc_sol1 = copy.deepcopy(doc)
    doc_sol1.sentences[0].tokens[3].text ="vacancy"
    doc_sol1.sentences[0].tokens[4].text ="to"
    doc_sol1.sentences[1].tokens[5].text ="vacancy"
    doc_sol1.sentences[1].tokens[6].text ="to"

    # test2 - tests case2 (more or equal number of tags) (Parameters: num_of_words=3, kind_of_word=['NN', 'IN', 'DT', 'NN'])
    doc_to_aug2 = copy.deepcopy(doc)
    doc_sol2 = copy.deepcopy(doc)
    doc_sol2.sentences[0].tokens[3].text = "vacancy"
    doc_sol2.sentences[0].tokens[4].text = "to"
    doc_sol2.sentences[0].tokens[5].text = "the"
    doc_sol2.sentences[1].tokens[13].text = "Head"
    doc_sol2.sentences[1].tokens[14].text = "of"
    doc_sol2.sentences[1].tokens[15].text = "the"

    # test3 - tests case3 (fewer tags are given) (Parameters: num_of_words=3, kind_of_word=['NN', 'IN'])
    doc_to_aug3 = copy.deepcopy(doc)
    doc_sol3 = copy.deepcopy(doc)
    doc_sol3.sentences[0].tokens[3].text = "vacancy"
    doc_sol3.sentences[0].tokens[4].text = "to"
    doc_sol3.sentences[0].tokens[5].text = "the"

    # ACT
    # test1
    first_token1 = Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="O", sentence_index=0)
    first_token2 = Token(text='number', index_in_document=5, pos_tag='NN', bio_tag='I-Activity Data', sentence_index=1)
    possible_seq1 = [['number', 'of'], ['vacancy', 'to']]
    possible_seq2 = [['vacancy', 'to'], ['vacancy', 'to']]
    mockito.when(trafo103).choice(sentence1.tokens).thenReturn(first_token1)
    mockito.when(trafo103).choice(sentence2.tokens).thenReturn(first_token2)
    mockito.when(trafo103).choice(possible_seq1).thenReturn(['vacancy', 'to'])
    mockito.when(trafo103).choice(possible_seq2).thenReturn(['vacancy', 'to'])

    doc_aug1 = trafo1.do_augment(doc_to_aug1)

    # test2
    possible_seq3 = [['vacancy', 'to', 'the']]
    possible_seq4 = [['Head', 'of', 'the']]
    mockito.when(trafo103).choice(possible_seq3).thenReturn(['vacancy', 'to', 'the'])
    mockito.when(trafo103).choice(possible_seq4).thenReturn(['Head', 'of', 'the'])

    doc_aug2 = trafo2.do_augment(doc_to_aug2)

    # test3
    possible_seq5 = [['number', 'of', 'the']]
    mockito.when(trafo103).choice(possible_seq3).thenReturn(['vacancy', 'to', 'the'])
    mockito.when(trafo103).choice(possible_seq5).thenReturn(['number', 'of', 'the'])
    doc_aug3 = trafo3.do_augment(doc_to_aug3)


    # ASSERT
    assert doc_aug1 == doc_sol1
    assert doc_aug2 == doc_sol2
    assert doc_aug3 == doc_sol3