import random
import string

import typing
import uuid

import data
from augment import trafo24
from data import model


def test_do_augment():
    sentences = [
        model.Sentence(
            tokens=[
                model.Token(
                    text="First",
                    index_in_document=0,
                    pos_tag="",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text="sentence",
                    index_in_document=1,
                    pos_tag="",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text="content",
                    index_in_document=2,
                    pos_tag="",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text=".",
                    index_in_document=3,
                    pos_tag="",
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
                    pos_tag="",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="short",
                    index_in_document=5,
                    pos_tag="",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="sentence",
                    index_in_document=6,
                    pos_tag="",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="!",
                    index_in_document=7,
                    pos_tag="",
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
                    pos_tag="",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="and",
                    index_in_document=9,
                    pos_tag="",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="last",
                    index_in_document=10,
                    pos_tag="",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="!",
                    index_in_document=11,
                    pos_tag="",
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

    trafo = trafo24.Trafo24Step(dataset=[document], n=1)

    augmented = trafo.do_augment(document)

    assert augmented != document
    assert len(augmented.sentences) == 2
    assert len(augmented.tokens) == len(document.tokens) - 1
    for mention in augmented.mentions:
        assert mention.sentence_index < len(augmented.sentences)
        assert all([i < len(augmented.tokens) for i in mention.token_indices])

    augmented = trafo.do_augment(augmented)

    assert len(augmented.sentences) == 1
    assert all([t.sentence_index == 0 for t in augmented.sentences[0].tokens])
    for mention in augmented.mentions:
        assert mention.sentence_index < len(augmented.sentences)
        assert all([i < len(augmented.tokens) for i in mention.token_indices])

    assert augmented.mentions[0].text(augmented) == "sentence"
    assert augmented.mentions[1].text(augmented) == "Second short sentence"

    augmented = document.copy()
    trafo.merge_sentences(0, augmented)

    assert set([t.sentence_index for t in augmented.tokens]) == {0, 1}
    for mention in augmented.mentions:
        assert mention.sentence_index < len(augmented.sentences)
        assert all([i < len(augmented.tokens) for i in mention.token_indices])


def test_trafo24_monkey():
    def random_mentions(sentences: typing.List[model.Sentence]):
        mentions = []
        for _ in range(20):
            sentence_id = random.randrange(0, len(sentences))
            mention_len = random.randint(1, min(len(sentences[sentence_id].tokens) - 1, 4))
            indices_start = random.randint(
                0, len(sentences[sentence_id].tokens) - mention_len - 1
            )
            indices = []
            for i in range(mention_len):
                indices.append(i + indices_start)
            mentions.append(
                data.Mention(
                    ner_tag=uuid.uuid4().__str__(),
                    sentence_index=sentence_id,
                    token_indices=indices,
                )
            )
        return mentions

    def random_sentence(sentence_id: int, num_previous_tokens: int):
        tokens = []
        for i in range(0, random.randint(1, 4)):
            tokens.append(random_token(sentence_id, num_previous_tokens + i))
        tokens.append(
            model.Token(
                text=".",
                index_in_document=num_previous_tokens + len(tokens),
                pos_tag="",
                bio_tag="",
                sentence_index=sentence_id,
            )
        )
        return model.Sentence(tokens=tokens)

    def random_token(sentence_id: int, id_in_doc) -> model.Token:
        return model.Token(
            text="".join(random.choice(string.ascii_lowercase) for _ in range(3)),
            sentence_index=sentence_id,
            index_in_document=id_in_doc,
            bio_tag="",
            pos_tag="",
        )

    def random_doc():
        sentences = []
        tokens_in_doc = 0
        for sent_id in range(3):
            sentence = random_sentence(sent_id, tokens_in_doc)
            tokens_in_doc += len(sentence.tokens)
            sentences.append(sentence)
        mentions = random_mentions(sentences)
        return model.Document(
            text="",
            name="",
            sentences=sentences,
            mentions=mentions,
            entities=[],
            relations=[],
        )

    for i in range(10):
        doc = random_doc()
        trafo = trafo24.Trafo24Step(
            dataset=[doc], n=random.randint(1, len(doc.sentences))
        )

        augmented = trafo.do_augment(doc)

        assert augmented != doc
        for mention in augmented.mentions:
            assert mention.sentence_index < len(augmented.sentences)
            assert all([i < len(augmented.tokens) for i in mention.token_indices])

        for old_mention, new_mention in zip(doc.mentions, augmented.mentions):
            assert old_mention.ner_tag == new_mention.ner_tag
            assert old_mention.text(doc) == new_mention.text(augmented)
