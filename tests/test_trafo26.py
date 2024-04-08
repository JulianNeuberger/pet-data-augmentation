import pos_enum
from augment import trafo26
from data import model


def test_do_augment():
    sentences = [
        model.Sentence(
            tokens=[
                model.Token(
                    text="First",
                    index_in_document=0,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text="sentence",
                    index_in_document=1,
                    pos_tag="NN",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text="content",
                    index_in_document=2,
                    pos_tag="NN",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text=".",
                    index_in_document=3,
                    pos_tag=".",
                    bio_tag="",
                    sentence_index=0,
                ),
            ]
        ),
        model.Sentence(
            tokens=[
                model.Token(
                    text="Second",
                    index_in_document=4,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="short",
                    index_in_document=5,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="sentence",
                    index_in_document=6,
                    pos_tag="NN",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="!",
                    index_in_document=7,
                    pos_tag=".",
                    bio_tag="",
                    sentence_index=1,
                ),
            ]
        ),
        model.Sentence(
            tokens=[
                model.Token(
                    text="Third",
                    index_in_document=8,
                    pos_tag="NNP",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="and",
                    index_in_document=9,
                    pos_tag="CC",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="last",
                    index_in_document=10,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="!",
                    index_in_document=11,
                    pos_tag=".",
                    bio_tag="",
                    sentence_index=2,
                ),
            ]
        ),
    ]

    mentions = [
        model.Mention(ner_tag="A", sentence_index=0, token_indices=[1]),
        model.Mention(ner_tag="B", sentence_index=1, token_indices=[0, 1, 2]),
    ]

    entities = [model.Entity(mention_indices=[0]), model.Entity(mention_indices=[1])]

    relations = [
        model.Relation(
            head_entity_index=0, tail_entity_index=1, tag="RR", evidence=[0, 1]
        )
    ]

    document = model.Document(
        name="",
        sentences=sentences,
        mentions=mentions,
        entities=entities,
        relations=relations,
        text="",
    )

    trafo = trafo26.Trafo26Step(dataset=[document], n=1, tag_groups=[pos_enum.Pos.NOUN, pos_enum.Pos.VERB])

    augmented = trafo.do_augment(document)

    assert augmented != document


