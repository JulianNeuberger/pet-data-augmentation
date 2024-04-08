import pytest

from transformations import tokenmanager
from data import model
import copy


def document_fixture():
    doc = model.Document(
        text="",
        name="test",
        sentences=[
            model.Sentence(tokens=[
                model.Token("This", 0, "A", "O", 0),
                model.Token("is", 1, "B", "O", 0),
                model.Token("a", 2, "C", "O", 0),
                model.Token("sentence", 3, "A", "B-Object", 0),
                model.Token(".", 4, "A", "O", 0),
            ]),
            model.Sentence(tokens=[
                model.Token("And", 5, "A", "O", 1),
                model.Token("another", 6, "B", "B-Object", 1),
                model.Token("one", 7, "C", "I-Object", 1),
                model.Token("!", 8, "A", "0", 1),
            ]),
            model.Sentence(tokens=[
                model.Token("Here", 9, "A", "B-Something", 2),
                model.Token("comes", 10, "B", "B-Activity", 2),
                model.Token("the", 11, "C", "O", 2),
                model.Token("last", 12, "C", "O", 2),
                model.Token("one", 13, "C", "B-Object", 2),
                model.Token(".", 14, "A", "0", 2),
            ]),
        ],
        mentions=[
            model.Mention("Object", 0, [3]),
            model.Mention("Object", 1, [1, 2]),
            model.Mention("Something", 2, [0]),
            model.Mention("Activity", 2, [1]),
            model.Mention("Object", 2, [4]),
        ],
        entities=[
            model.Entity([0, 1]),
            model.Entity([2]),
            model.Entity([3]),
            model.Entity([4]),
        ],
        relations=[
            model.Relation(0, 2, "Testtag", [0, 2]),
            model.Relation(1, 2, "Othertag", [1, 2]),
            model.Relation(3, 2, "Lasttag", [2]),
        ]
    )
    return doc


def test_get_pos_tag():
    # ARRANGE
    p1 = []
    p2 = ["my"]
    p3 = ["my", "name", "is"]

    # ACT
    t1 = tokenmanager.get_pos_tag(p1)
    t2 = tokenmanager.get_pos_tag(p2)
    t3 = tokenmanager.get_pos_tag(p3)

    # ASSERT
    assert t1 == []
    assert t2 == ["PRP$"]
    assert t3 == ["PRP$", "NN", "VBZ"]


def test_get_bio_tag_based_on_left_token():
    # ARRANGE
    p1 = "O"
    p2 = "B-Actor"
    p3 = "I-Actor"

    # ACT
    t1 = tokenmanager.get_continued_bio_tag(p1)
    t2 = tokenmanager.get_continued_bio_tag(p2)
    t3 = tokenmanager.get_continued_bio_tag(p3)

    # ASSERT
    assert t1 == p1
    assert t2 == p3
    assert t3 == p3


def test_get_index_in_sentence():
    # ARRANGE
    # test1 - sentence does not exist
    sent1 = model.Sentence(tokens=[])
    text1 = ["I", "am"]
    ind_in_doc1 = 0

    # test2 - text does not exist
    toks2 = []
    toks2.append(
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        )
    )
    sent2 = model.Sentence(tokens=toks2)
    text2 = []
    ind_in_doc2 = 0

    # test3 - both exist, text is not in sentence
    toks3 = []
    toks3.append(
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        )
    )
    toks3.append(
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        )
    )
    toks3.append(
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        )
    )
    toks3.append(
        model.Token(
            text="I", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        )
    )
    sent3 = model.Sentence(tokens=toks3)
    text3 = ["am", "the", "worst"]
    ind_in_doc3 = 0

    # test4 - both exist, text is in sentence
    sent4 = model.Sentence(tokens=toks3)
    text4 = ["am", "the"]
    ind_in_doc4 = 1

    # test5 - both exist, text is twice in sentence
    sent5 = copy.deepcopy(sent4)
    sent5.tokens.append(
        model.Token(
            text="am", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        )
    )
    sent5.tokens.append(
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        )
    )
    text5 = ["am", "the"]
    ind_in_doc5 = 4

    # Act
    i1 = tokenmanager.get_index_in_sentence(sent1, text1, ind_in_doc1)
    i2 = tokenmanager.get_index_in_sentence(sent2, text2, ind_in_doc2)
    i3 = tokenmanager.get_index_in_sentence(sent3, text3, ind_in_doc3)
    i4 = tokenmanager.get_index_in_sentence(sent4, text4, ind_in_doc4)
    i5 = tokenmanager.get_index_in_sentence(sent5, text5, ind_in_doc5)

    # Assert
    assert i1 is None
    assert i2 is None
    assert i3 is None
    assert i4 == 1
    assert i5 == 4


def test_get_mentions():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="chief", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    mention = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])

    doc = model.Document(
        text="I am the chief of the department.",
        name="1",
        sentences=[sentence],
        mentions=[mention],
        entities=[],
        relations=[],
    )

    # test1 - token is not in mentions
    ind_in_sent1 = 5
    sent_ind1 = 0

    # test2 - token is in mentions
    ind_in_sent2 = 2
    sent_ind2 = 0

    # ACT
    ment1 = tokenmanager.get_mentions(doc, ind_in_sent1, sent_ind1)
    ment2 = tokenmanager.get_mentions(doc, ind_in_sent2, sent_ind2)

    # ASSERT
    assert ment1 == []
    assert ment2 == [0]


def test_insert_token_in_mentions():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="chief", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]

    sentence = model.Sentence(tokens=tokens)

    mention1 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention2 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3, 5])

    doc = model.Document(
        text="I am the chief of the department.",
        name="1",
        sentences=[sentence],
        mentions=[mention1],
        entities=[],
        relations=[],
    )
    doc2 = model.Document(
        text="I am the chief of the department.",
        name="1",
        sentences=[sentence],
        mentions=[mention2],
        entities=[],
        relations=[],
    )

    # test1 - mention_id does not exist (index_in_sentence is always ok due to the prior method)
    ment_id1 = 2
    index_in_sent1 = 5
    doc_to_aug1 = copy.deepcopy(doc)
    doc_sol1 = copy.deepcopy(doc)

    # test2 - mention_id exists
    ment_id2 = 0
    index_in_sent2 = 5
    doc_to_aug2 = copy.deepcopy(doc)
    doc_sol2 = copy.deepcopy(doc2)

    # test3 - no duplicate index_in_sentence_id
    ment_id3 = 0
    index_in_sent3 = 5
    doc_to_aug3 = copy.deepcopy(doc2)
    doc_sol3 = copy.deepcopy(doc2)

    # ACT
    # test1
    tokenmanager.insert_token_in_mentions(doc_to_aug1, index_in_sent1, ment_id1)

    # test2
    tokenmanager.insert_token_in_mentions(doc_to_aug2, index_in_sent2, ment_id2)

    # test3
    tokenmanager.insert_token_in_mentions(doc_to_aug3, index_in_sent3, ment_id3)

    # ASSERT
    assert doc_to_aug1 == doc_sol1
    assert doc_to_aug2 == doc_sol2
    assert doc_to_aug3 == doc_sol3


def test_insert_token_in_tokens():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="chief", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]
    tokens1 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="can", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=11,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=12, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens2 = [
        model.Token(
            text="The", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    sentence1 = model.Sentence(tokens=tokens1)
    sentence2 = model.Sentence(tokens=tokens2)

    mentions1 = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]

    doc = model.Document(
        text="I am the chief of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence1, sentence2],
        mentions=mentions1,
        entities=[],
        relations=[],
    )

    # test1 - index in sentence is not correct
    tok1 = model.Token(
        text="The", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
    )
    ind_in_sent1 = -1
    doc_to_aug1 = copy.deepcopy(doc)
    doc_sol1 = copy.deepcopy(doc)

    # test2 - token.sentence_index is not correct
    tok2 = model.Token(
        text="The", index_in_document=0, pos_tag="", bio_tag="", sentence_index=3
    )
    ind_in_sent2 = 1
    doc_to_aug2 = copy.deepcopy(doc)
    doc_sol2 = copy.deepcopy(doc)

    # test3 - both are correct, token hast to be inserted
    tok3 = model.Token(
        text="Hallo", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
    )
    ind_in_sent3 = 1
    doc_to_aug3 = copy.deepcopy(doc)

    tokens3 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        tok3,
        model.Token(
            text="can", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=11, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=12,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=13, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens4 = [
        model.Token(
            text="The", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=18, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]

    sentence3 = model.Sentence(tokens=tokens3)
    sentence4 = model.Sentence(tokens=tokens4)

    mentions2 = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 2]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]

    doc_sol3 = model.Document(
        text="I am the chief of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence3, sentence4],
        mentions=mentions2,
        entities=[],
        relations=[],
    )

    # ACT
    # test1
    tokenmanager.insert_token_in_tokens(doc_to_aug1, tok1, ind_in_sent1)

    # test2
    tokenmanager.insert_token_in_tokens(doc_to_aug2, tok2, ind_in_sent2)

    # test3
    tokenmanager.insert_token_in_tokens(doc_to_aug3, tok3, ind_in_sent3)

    # ASSERT
    assert doc_to_aug1 == doc_sol1
    assert doc_to_aug2 == doc_sol2
    assert doc_to_aug3 == doc_sol3


def test_create_token():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="chief", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]
    tokens1 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="can", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=11,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=12, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens2 = [
        model.Token(
            text="The", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    sentence1 = model.Sentence(tokens=tokens1)
    sentence2 = model.Sentence(tokens=tokens2)

    mentions = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]

    doc = model.Document(
        text="I am the chief of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence1, sentence2],
        mentions=mentions,
        entities=[],
        relations=[],
    )

    # test1 - index in sentence is not valid
    ind_in_sent1 = 8
    tok = model.Token(
        text="Hallo", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
    )
    ment_ind1 = None
    doc_to_augment1 = copy.deepcopy(doc)

    # test2 - mention id is not valid
    ind_in_sent2 = 2
    ment_ind2 = 5
    doc_to_augment2 = copy.deepcopy(doc)

    # test3 - both ids are valid, don't change mention
    ind_in_sent3 = 1
    doc_to_augment3 = copy.deepcopy(doc)

    tokens3 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        tok,
        model.Token(
            text="can", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=11, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=12,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=13, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens4 = [
        model.Token(
            text="The", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=18, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]

    sentence3 = model.Sentence(tokens=tokens3)
    sentence4 = model.Sentence(tokens=tokens4)

    mentions1 = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 2]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]

    doc_sol3 = model.Document(
        text="I am the chief of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence3, sentence4],
        mentions=mentions1,
        entities=[],
        relations=[],
    )

    # test4 - both are valid, don't change mention
    ind_in_sent4 = 1
    ment_ind4 = 1
    doc_to_augment4 = copy.deepcopy(doc)
    mentions2 = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1, 2]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]

    doc_sol4 = model.Document(
        text="I am the chief of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence3, sentence4],
        mentions=mentions2,
        entities=[],
        relations=[],
    )

    # ACT
    # test1
    tokenmanager.create_token(doc_to_augment1, tok, ind_in_sent1, ment_ind1)

    # test2
    tokenmanager.create_token(doc_to_augment2, tok, ind_in_sent2, ment_ind2)

    # test3
    tokenmanager.create_token(doc_to_augment3, tok, ind_in_sent3, None)

    # test4
    tokenmanager.create_token(doc_to_augment4, tok, ind_in_sent4, ment_ind4)

    # ASSERT
    assert doc_to_augment1 == doc
    assert doc_to_augment2 == doc
    assert doc_to_augment3 == doc_sol3
    assert doc_to_augment4 == doc_sol4


def test_delete_token_from_tokens():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="head", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]
    tokens1 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="can", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=11,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=12, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens2 = [
        model.Token(
            text="The", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    sentence1 = model.Sentence(tokens=tokens1)
    sentence2 = model.Sentence(tokens=tokens2)

    mentions = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]

    doc = model.Document(
        text="I am the head of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence1, sentence2],
        mentions=mentions,
        entities=[],
        relations=[],
    )

    # test1 - index_in_document does not exist
    ind_in_doc1 = 20
    doc_sol1 = copy.deepcopy(doc)
    doc_to_aug1 = copy.deepcopy(doc)

    # test2 - index_in_document does exist
    ind_in_doc2 = 10
    doc_to_aug2 = copy.deepcopy(doc)
    doc_sol2 = model.Document(
        text="I am the head of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence1, sentence2],
        mentions=mentions,
        entities=[],
        relations=[],
    )
    doc_sol2.sentences[1].tokens.pop(2)
    doc_sol2.sentences[1].tokens[2].index_in_document = 10
    doc_sol2.sentences[1].tokens[3].index_in_document = 11
    doc_sol2.sentences[2].tokens[0].index_in_document = 12
    doc_sol2.sentences[2].tokens[1].index_in_document = 13
    doc_sol2.sentences[2].tokens[2].index_in_document = 14
    doc_sol2.sentences[2].tokens[3].index_in_document = 15
    doc_sol2.sentences[2].tokens[4].index_in_document = 16

    # ACT
    # test1
    with pytest.raises(IndexError):
        tokenmanager.delete_token_from_tokens(doc_to_aug1, ind_in_doc1)

    # test2
    index2 = tokenmanager.delete_token_from_tokens(doc_to_aug2, ind_in_doc2)

    # ASSERT
    assert doc_to_aug1 == doc_sol1

    assert doc_to_aug2 == doc_sol2
    assert index2 == (1, 2)

    assert tuple([t.index_in_document for t in doc_to_aug2.tokens]) == tuple(
        range(len(doc_to_aug2.tokens))
    )


def test_delete_token_from_mention_token_indices():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="head", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]
    tokens1 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="can", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=11,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=12, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens2 = [
        model.Token(
            text="The", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    sentence1 = model.Sentence(tokens=tokens1)
    sentence2 = model.Sentence(tokens=tokens2)

    mentions = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1, 2]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]

    doc = model.Document(
        text="I am the head of the department. I can do everything. The job is nice.",
        name="1",
        sentences=[sentence, sentence1, sentence2],
        mentions=mentions,
        entities=[],
        relations=[],
    )

    # test1
    ind_in_sent1 = 1
    sent_ind1 = 1

    doc_sol1 = copy.deepcopy(doc)
    mentions_new = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1, 2]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
    ]
    doc_sol1.mentions = mentions_new

    # ACT
    # test1
    doc_to_aug1 = copy.deepcopy(doc)
    mention_to_delete = tokenmanager.delete_token_from_mention_token_indices(
        doc_to_aug1, ind_in_sent1, sent_ind1
    )

    # ASSERT
    assert doc_to_aug1 == doc_sol1
    assert mention_to_delete == 1


def test_change_mention_indices_in_entities():
    # ARRANGE
    mentions = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1, 2]),
    ]

    entities = [
        model.Entity(mention_indices=[0, 1]),
        model.Entity(mention_indices=[1]),
        model.Entity(mention_indices=[2, 3, 4]),
    ]

    doc = model.Document(
        text="I ",
        name="1",
        sentences=[],
        mentions=mentions,
        entities=entities,
        relations=[],
    )

    # test1
    ment_id = 1
    doc_to_aug1 = copy.deepcopy(doc)

    doc_sol1 = copy.deepcopy(doc)
    doc_sol1.entities[2].mention_indices.pop()
    doc_sol1.entities[2].mention_indices.insert(0, 1)

    # ACT
    # test1
    tokenmanager.adjust_mention_indices_in_entities(doc_to_aug1, ment_id)

    # ASSERT
    assert doc_to_aug1 == doc_sol1


def test_delete_mention_from_entity():
    # ARRANGE
    mentions = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[1, 2]),
    ]

    entities = [
        model.Entity(mention_indices=[0, 1]),
        model.Entity(mention_indices=[1]),
        model.Entity(mention_indices=[2, 3, 4]),
    ]

    doc = model.Document(
        text="I ",
        name="1",
        sentences=[],
        mentions=mentions,
        entities=entities,
        relations=[],
    )

    # test1
    ment_ind = 1
    doc_to_aug1 = copy.deepcopy(doc)

    doc_sol1 = copy.deepcopy(doc)
    doc_sol1.entities.pop(1)
    doc_sol1.entities[0].mention_indices.pop()

    # ACT
    # test1
    tokenmanager.delete_mention_from_entities(doc_to_aug1, ment_ind)

    # ASSERT
    assert doc_to_aug1 == doc_sol1


def test_delete_relations():
    # ARRANGE
    entity_index = 2
    relations = [
        model.Relation(head_entity_index=2, tail_entity_index=3, tag="", evidence=[]),
        model.Relation(head_entity_index=1, tail_entity_index=2, tag="", evidence=[]),
        model.Relation(head_entity_index=3, tail_entity_index=4, tag="", evidence=[]),
    ]

    doc = model.Document(
        text="I ", name="1", sentences=[], mentions=[], entities=[], relations=relations
    )

    # test1
    doc_to_aug1 = copy.deepcopy(doc)

    doc_sol1 = copy.deepcopy(doc)
    doc_sol1.relations.pop(0)
    doc_sol1.relations.pop(0)

    # ACT
    # test1
    tokenmanager.delete_relations(doc_to_aug1, entity_index)

    # ASSERT
    assert doc_to_aug1 == doc_sol1


def test_delete_sentence():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="chief", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]
    tokens1 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="can", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=11,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=12, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens2 = [
        model.Token(
            text="The", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]
    tokens3 = [
        model.Token(
            text="I", index_in_document=18, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text="like", index_in_document=19, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text="the", index_in_document=20, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text="job", index_in_document=21, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text=".", index_in_document=22, pos_tag="", bio_tag="", sentence_index=3
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    sentence1 = model.Sentence(tokens=tokens1)
    sentence2 = model.Sentence(tokens=tokens2)
    sentence3 = model.Sentence(tokens=tokens3)

    mentions = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
        model.Mention(ner_tag="Actor", sentence_index=3, token_indices=[1, 2]),
    ]

    doc = model.Document(
        text="I am the chief of the department. I can do everything. The job is nice. I like the job.",
        name="1",
        sentences=[sentence, sentence1, sentence2, sentence3],
        mentions=mentions,
        entities=[],
        relations=[],
    )

    # test1 - delete second sentence
    doc_to_aug1 = copy.deepcopy(doc)
    doc_sol1 = copy.deepcopy(doc)
    sent_index1 = 1

    del doc_sol1.sentences[sent_index1]

    tokens4 = [
        model.Token(
            text="The", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="job", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="is", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="nice", index_in_document=11, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text=".", index_in_document=12, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens5 = [
        model.Token(
            text="I", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="like", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="the", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]
    mentions2 = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[4]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[1, 2]),
    ]

    doc_sol1.sentences[1].tokens = tokens4
    doc_sol1.sentences[2].tokens = tokens5
    doc_sol1.mentions = mentions2

    # ACT
    # test1
    tokenmanager.delete_sentence(doc_to_aug1, sent_index1)

    # ASSERT
    assert doc_to_aug1 == doc_sol1


def test_replace_mention_text():
    # ARRANGE
    tokens = [
        model.Token(
            text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="am", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="chief", index_in_document=3, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="of", index_in_document=4, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="the", index_in_document=5, pos_tag="", bio_tag="", sentence_index=0
        ),
        model.Token(
            text="department",
            index_in_document=6,
            pos_tag="",
            bio_tag="",
            sentence_index=0,
        ),
        model.Token(
            text=".", index_in_document=7, pos_tag="", bio_tag="", sentence_index=0
        ),
    ]
    tokens1 = [
        model.Token(
            text="I", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="can", index_in_document=9, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="do", index_in_document=10, pos_tag="", bio_tag="", sentence_index=1
        ),
        model.Token(
            text="everything",
            index_in_document=11,
            pos_tag="",
            bio_tag="",
            sentence_index=1,
        ),
        model.Token(
            text=".", index_in_document=12, pos_tag="", bio_tag="", sentence_index=1
        ),
    ]
    tokens2 = [
        model.Token(
            text="The", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="job", index_in_document=14, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="is", index_in_document=15, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text="nice", index_in_document=16, pos_tag="", bio_tag="", sentence_index=2
        ),
        model.Token(
            text=".", index_in_document=17, pos_tag="", bio_tag="", sentence_index=2
        ),
    ]
    tokens3 = [
        model.Token(
            text="I", index_in_document=18, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text="like", index_in_document=19, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text="the", index_in_document=20, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text="job", index_in_document=21, pos_tag="", bio_tag="", sentence_index=3
        ),
        model.Token(
            text=".", index_in_document=22, pos_tag="", bio_tag="", sentence_index=3
        ),
    ]

    sentence = model.Sentence(tokens=tokens)
    sentence1 = model.Sentence(tokens=tokens1)
    sentence2 = model.Sentence(tokens=tokens2)
    sentence3 = model.Sentence(tokens=tokens3)

    mentions = [
        model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3]),
        model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0]),
        model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4]),
        model.Mention(ner_tag="Object", sentence_index=3, token_indices=[2, 3]),
    ]

    doc = model.Document(
        text="I am the chief of the department. I can do everything. The job is nice. I like the job.",
        name="1",
        sentences=[sentence, sentence1, sentence2, sentence3],
        mentions=mentions,
        entities=[],
        relations=[],
    )

    # same length
    tokenmanager.replace_mention_text(doc, 0, ["a", "boss"])
    assert doc.mentions[0].text(doc) == "a boss"

    # longer text
    tokenmanager.replace_mention_text(doc, 1, ["The", "boss"])
    assert doc.mentions[1].text(doc) == "The boss"
    assert (
        " ".join([t.text for t in doc.sentences[1].tokens])
        == "The boss can do everything ."
    )

    # shorter text
    tokenmanager.replace_mention_text(doc, 3, ["it"])
    assert doc.mentions[3].text(doc) == "it"
    assert " ".join([t.text for t in doc.sentences[3].tokens]) == "I like it ."


def test_replace_sequence_text_in_sentence():
    doc = document_fixture()
    tokenmanager.replace_sequence_text_in_sentence(doc, 0, 3, 4, ["sent"])
    assert doc.sentences[0].text == "This is a sent ."
    assert doc.mentions[0].text(doc) == "sent"

    doc = document_fixture()
    tokenmanager.replace_sequence_text_in_sentence(doc, 0, 3, 5, ["sent"])
    assert doc.sentences[0].text == "This is a sent"
    assert doc.mentions[0].text(doc) == "sent"

    doc = document_fixture()
    tokenmanager.replace_sequence_text_in_sentence(doc, 0, 3, 4, ["example", "sent"])
    assert doc.sentences[0].text == "This is a example sent ."
    assert doc.mentions[0].text(doc) == "example sent"
