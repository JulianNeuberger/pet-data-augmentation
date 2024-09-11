from data import mutate

from tests.common import document_fixture


def test_swap_sentences():
    doc = document_fixture()
    mutate.swap_sentences_inplace(doc, 0, 2)
    assert doc.tokens[0].text == "Here"
    assert doc.tokens[6].text == "And"
    assert doc.tokens[10].text == "This"
    assert doc.mentions[0].token_document_indices == tuple([13])
    assert doc.mentions[4].token_document_indices == tuple([4])


def test_insert_tokens():
    doc = document_fixture()
    assert doc.tokens[0].text == "Here"
    mutate.insert_tokens_inplace(doc)
