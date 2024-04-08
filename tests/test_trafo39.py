import copy
from augment.trafo39 import Trafo39Step
from augment import trafo39
import mockito
from data.model import Token, Sentence, Document, Entity, Mention

# Author for entire script: Leonie
def test_do_augment():
    # ARRANGE
    # Trafo Objects for testing do_augment()
    trafo1 = Trafo39Step()


    # test1 - no entities (Parameters: mention_new = False)
    tokens1 = [Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="O", sentence_index=0),
               Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="O", sentence_index=0),
               Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="O", sentence_index=0),
               Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="O", sentence_index=0),
               Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="O", sentence_index=0),
               Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="O", sentence_index=0),
               Token(text="functional", index_in_document=6, pos_tag="JJ", bio_tag="O", sentence_index=0),
               Token(text="department", index_in_document=7, pos_tag="NN", bio_tag="O", sentence_index=0),
               Token(text=".", index_in_document=8, pos_tag=".", bio_tag="O", sentence_index=0)]

    sentence1 = Sentence(tokens1)

    doc1 = Document(
        text="I am the Head of the functional department.",
        name="1", sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[])
    doc_to_aug1 = copy.deepcopy(doc1)
    doc_sol1 = copy.deepcopy(doc1)

    # test2 (Parameters: mention_new = False)
    sentence2 = Sentence(tokens=[Token(text='When', index_in_document=0, pos_tag='WRB', bio_tag='O', sentence_index=0),
                                 Token(text='I', index_in_document=1, pos_tag='PRP', bio_tag='B-Actor',
                                       sentence_index=0),
                                 Token(text='have', index_in_document=2, pos_tag='VBP', bio_tag='O', sentence_index=1),
                                 Token(text='detected', index_in_document=3, pos_tag='VBN', bio_tag='B-Activity',
                                       sentence_index=0),
                                 Token(text='a', index_in_document=4, pos_tag='DT', bio_tag='B-Activity Data',
                                       sentence_index=0),
                                 Token(text='number', index_in_document=5, pos_tag='NN', bio_tag='I-Activity Data',
                                       sentence_index=0),
                                 Token(text='of', index_in_document=6, pos_tag='IN', bio_tag='I-Activity Data',
                                       sentence_index=0),
                                 Token(text='personnel', index_in_document=7, pos_tag='NN', bio_tag='I-Activity Data',
                                       sentence_index=0),
                                 Token(text='requirements', index_in_document=8, pos_tag='NNS',
                                       bio_tag='I-Activity Data',
                                       sentence_index=0),
                                 Token(text=',', index_in_document=9, pos_tag=',', bio_tag='O', sentence_index=0),
                                 Token(text='I', index_in_document=10, pos_tag='PRP', bio_tag='B-Actor',
                                       sentence_index=0),
                                 Token(text='report', index_in_document=11, pos_tag='VBP', bio_tag='B-Activity',
                                       sentence_index=0),
                                 Token(text='the', index_in_document=12, pos_tag='DT', bio_tag='B-Activity Data',
                                       sentence_index=0),
                                 Token(text='vacancy', index_in_document=13, pos_tag='NN', bio_tag='I-Activity Data',
                                       sentence_index=0),
                                 Token(text='to', index_in_document=14, pos_tag='IN', bio_tag='O', sentence_index=0),
                                 Token(text='the', index_in_document=15, pos_tag='DT', bio_tag='B-Actor',
                                       sentence_index=0),
                                 Token(text='Personnel', index_in_document=16, pos_tag='NNP', bio_tag='I-Actor',
                                       sentence_index=0),
                                 Token(text='Department', index_in_document=17, pos_tag='NNP', bio_tag='I-Actor',
                                       sentence_index=0),
                                 Token(text='.', index_in_document=18, pos_tag='.', bio_tag='O', sentence_index=0)])

    mentions2 = [Mention(ner_tag='Actor', sentence_index=0, token_indices=[1]),
                 Mention(ner_tag='Actor', sentence_index=0, token_indices=[10]),
                 Mention(ner_tag='Activity', sentence_index=0, token_indices=[3]),
                 Mention(ner_tag='Activity Data', sentence_index=0, token_indices=[4, 5, 6, 7, 8]),
                 Mention(ner_tag='Activity', sentence_index=0, token_indices=[11]),
                 Mention(ner_tag='Activity Data', sentence_index=0, token_indices=[12, 13]),
                 Mention(ner_tag='Actor', sentence_index=0, token_indices=[15, 16, 17])]

    entities2 = [Entity(mention_indices=[0, 1, 2, 3, 4]), Entity(mention_indices=[5]), Entity(mention_indices=[6]),
                 Entity(mention_indices=[7]), Entity(mention_indices=[8]), Entity(mention_indices=[9]),
                 Entity(mention_indices=[10]), Entity(mention_indices=[11, 12, 13]), Entity(mention_indices=[14]),
                 Entity(mention_indices=[15]), Entity(mention_indices=[16]), Entity(mention_indices=[17]),
                 Entity(mention_indices=[18]), Entity(mention_indices=[19])]

    doc2 = Document(
        text="When I have detected a number of personnel requirements , I report the vacancy to the Personnel Department .",
        name="1", sentences=[sentence2],
        mentions=mentions2, entities=entities2, relations=[])

    doc_to_aug2 = copy.deepcopy(doc2)
    doc_sol2 = copy.deepcopy(doc2)
    tokens3 = [Token(text='When', index_in_document=0, pos_tag='WRB', bio_tag='O', sentence_index=0),
                 Token(text='the', index_in_document=1, pos_tag='DT', bio_tag='B-Actor', sentence_index=0),
                 Token(text='Personnel', index_in_document=2, pos_tag='NNS', bio_tag='I-Actor', sentence_index=0),
                 Token(text='Department', index_in_document=3, pos_tag='NNP', bio_tag='I-Actor', sentence_index=0),
                 Token(text='have', index_in_document=4, pos_tag='VBP', bio_tag='O', sentence_index=1),
                 Token(text='detected', index_in_document=5, pos_tag='VBN', bio_tag='B-Activity', sentence_index=0),
                 Token(text='the', index_in_document=6, pos_tag='DT', bio_tag='B-Activity Data', sentence_index=0),
                 Token(text='vacancy', index_in_document=7, pos_tag='NN', bio_tag='I-Activity Data', sentence_index=0),
                 Token(text=',', index_in_document=8, pos_tag=',', bio_tag='O', sentence_index=0),
                 Token(text='I', index_in_document=9, pos_tag='PRP', bio_tag='B-Actor', sentence_index=0),
                 Token(text='detected', index_in_document=10, pos_tag='VBN', bio_tag='B-Activity', sentence_index=0),
                 Token(text='the', index_in_document=11, pos_tag='DT', bio_tag='B-Activity Data', sentence_index=0),
                 Token(text='vacancy', index_in_document=12, pos_tag='NN', bio_tag='I-Activity Data', sentence_index=0),
                 Token(text='to', index_in_document=13, pos_tag='IN', bio_tag='O', sentence_index=0),
                 Token(text='the', index_in_document=14, pos_tag='DT', bio_tag='B-Actor', sentence_index=0),
                 Token(text='Personnel', index_in_document=15, pos_tag='NNP', bio_tag='I-Actor', sentence_index=0),
                 Token(text='Department', index_in_document=16, pos_tag='NNP', bio_tag='I-Actor', sentence_index=0),
                 Token(text='.', index_in_document=17, pos_tag='.', bio_tag='O', sentence_index=0)]
    sentence3 = Sentence(tokens3)
    doc_sol2.sentences[0] = sentence3

    mentions3 = [Mention(ner_tag='Actor', sentence_index=0, token_indices=[1, 2, 3]),
                 Mention(ner_tag='Actor', sentence_index=0, token_indices=[9]),
                 Mention(ner_tag='Activity', sentence_index=0, token_indices=[5]),
                 Mention(ner_tag='Activity Data', sentence_index=0, token_indices=[6, 7]),
                 Mention(ner_tag='Activity', sentence_index=0, token_indices=[10]),
                 Mention(ner_tag='Activity Data', sentence_index=0, token_indices=[11, 12]),
                 Mention(ner_tag='Actor', sentence_index=0, token_indices=[14, 15, 16])]
    doc_sol2.mentions = mentions3

    # ACT
    # test1
    doc_aug1 = trafo1.do_augment(doc_to_aug1)

    # test2
    entity_mentions_by_type_actor = ['I', 'the Personnel Department']
    entity_mentions_by_type_actor2 = ['the Personnel Department', 'I']
    entity_mentions_by_type_activity = ['detected', 'report']
    entity_mentions_by_type_activity2 = ['report', 'detected']
    entity_mentions_by_type_activity_data = ['a number of personnel requirements', 'the vacancy']
    entity_mentions_by_type_activity_data2 = ['the vacancy', 'a number of personnel requirements']

    mockito.when(trafo39).rand().thenReturn(0.2).thenReturn(1).thenReturn(0.2).thenReturn(1).thenReturn(
        0.2).thenReturn(1).thenReturn(1)

    mockito.when(trafo39).choice(entity_mentions_by_type_actor).thenReturn('the Personnel Department')
    mockito.when(trafo39).choice(entity_mentions_by_type_actor2).thenReturn('the Personnel Department')
    mockito.when(trafo39).choice(entity_mentions_by_type_activity_data).thenReturn('the vacancy')
    mockito.when(trafo39).choice(entity_mentions_by_type_activity_data2).thenReturn('the vacancy')
    mockito.when(trafo39).choice(entity_mentions_by_type_activity).thenReturn('detected')
    mockito.when(trafo39).choice(entity_mentions_by_type_activity2).thenReturn('detected')

    doc_aug2 = trafo1.do_augment(doc_to_aug2)

    # ASSERT
    assert doc_aug1 == doc_sol1
    assert doc_aug2 == doc_sol2
