from augment import trafo106
from data import Document, Sentence, Token, Mention


def get_doc():
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

    mentions = [
        Mention(ner_tag='Actor', sentence_index=0, token_indices=[0]),
        Mention(ner_tag='Actor', sentence_index=0, token_indices=[3, 4, 5, 6, 7]),
        Mention(ner_tag='Activity', sentence_index=1, token_indices=[3]),
        Mention(ner_tag='Activity', sentence_index=1, token_indices=[11])
    ]

    sentence1 = Sentence(tokens1)
    sentence2 = Sentence(tokens2)
    doc = Document(
        text="I am the Head of the functional department. When I have detected a number of personnel requirements , I report the vacancy to the Personnel Department .",
        name="1", sentences=[sentence1, sentence2], mentions=mentions, entities=[], relations=[])
    return doc


def test_do_augment():
    trafo = trafo106.Trafo106Step(n=10)
    doc = get_doc()
    augmented_doc = trafo.do_augment(doc)

    # assert a copy was made
    assert augmented_doc != doc

    print()
    print()
    print(" ".join([t.text for t in doc.tokens]))
    print(" ".join([t.text for t in augmented_doc.tokens]))
    print("----")

    # assert a replacement has been made
    assert any([l.text != r.text for l, r in zip(doc.tokens, augmented_doc.tokens)])


def test_truncation():
    trafo = trafo106.Trafo106Step(n=10)
    doc = get_doc()
    doc.sentences += doc.sentences * 100
    augmented_doc = trafo.do_augment(doc)
