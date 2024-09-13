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


def test_replace_sequence_inplace_same_length():
    doc = document_fixture()
    doc.mentions.pop(1)
    assert doc.get_mention_for_token(doc.tokens[0]) is None
    assert doc.get_mention_for_token(doc.tokens[1]) is None
    assert doc.get_mention_for_token(doc.tokens[2]) is None
    assert doc.get_mention_for_token(doc.tokens[3]) == doc.mentions[0]

    assert doc.mentions[0].token_document_indices == (3,)

    mutate.replace_sequence_inplace(doc, 1, 3, ["was", "the"])

    assert doc.tokens[0].text == "This"
    assert doc.tokens[1].text == "was"
    assert doc.tokens[2].text == "the"
    assert doc.tokens[3].text == "sentence"

    assert doc.mentions[0].token_document_indices == (3,)


def test_replace_sequence_inplace_shorter():
    doc = document_fixture()
    doc.mentions.pop(1)
    assert doc.get_mention_for_token(doc.tokens[0]) is None
    assert doc.get_mention_for_token(doc.tokens[1]) is None
    assert doc.get_mention_for_token(doc.tokens[2]) is None
    assert doc.get_mention_for_token(doc.tokens[3]) == doc.mentions[0]

    assert doc.mentions[0].token_document_indices == (3,)

    mutate.replace_sequence_inplace(doc, 0, 3, ["were"])

    assert doc.tokens[0].text == "were"
    assert doc.tokens[1].text == "sentence"

    assert doc.mentions[0].token_document_indices == (1,)
