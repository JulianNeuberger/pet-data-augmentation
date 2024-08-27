import typing

import nltk

from data import PetDocument, PetToken, PetEntity, PetMention


def insert_tokens_inplace(
    doc: PetDocument, tokens: typing.List[PetToken], index_in_document: int
) -> None:
    # first, adjust all mention token indices that are affected
    for m_id, m in enumerate(doc.mentions):
        new_token_ids = []
        affected = False
        # will be true, if the inserted sequence is inside a mention or "touches" it on the right side
        expanded = (
            min(m.token_document_indices)
            < index_in_document
            <= max(m.token_document_indices)
        )
        for token_id in m.token_document_indices:
            if token_id < index_in_document:
                new_token_ids.append(token_id)
            else:
                affected = True
                new_token_ids.append(token_id + len(tokens))
        if affected:
            if expanded:
                # inserted tokens into a mention!
                new_token_ids += list(
                    range(index_in_document, index_in_document + len(tokens))
                )
                new_token_ids = sorted(new_token_ids)
            doc.mentions[m_id] = PetMention(
                type=m.type, token_document_indices=tuple(new_token_ids)
            )

    for token in reversed(tokens):
        doc.tokens.insert(index_in_document, token)

    # adjust all following tokens
    for i in range(index_in_document + len(tokens), len(doc.tokens)):
        t = doc.tokens[i]
        doc.tokens[i] = PetToken(
            text=t.text,
            pos_tag=t.pos_tag,
            sentence_index=t.sentence_index,
            index_in_document=t.index_in_document + len(tokens),
        )


def insert_texts_inplace(
    document: PetDocument, texts: typing.List[str], index_in_document: int
) -> None:
    tokens = text_to_tokens(document, texts, index_in_document)
    insert_tokens_inplace(document, tokens, index_in_document)


def swap_sentences_inplace(
    document: PetDocument, first_sentence_id: int, second_sentence_id: int
) -> None:
    assert first_sentence_id != second_sentence_id

    if first_sentence_id > second_sentence_id:
        second_sentence_id, first_sentence_id = first_sentence_id, second_sentence_id

    first_sentence = document.sentences[first_sentence_id]
    second_sentence = document.sentences[second_sentence_id]

    sentence_id_map = {i: i for i in range(len(document.sentences))}
    sentence_id_map[first_sentence_id] = second_sentence_id
    sentence_id_map[second_sentence_id] = first_sentence_id

    # how many tokens will we shift left / right when we swap sentences?
    offset = len(first_sentence) - len(second_sentence)

    # start unchanged
    token_index_mapping = {
        t.index_in_document: t.index_in_document for t in document.tokens
    }

    first_sentence_start = first_sentence[0].index_in_document
    first_sentence_end = first_sentence[-1].index_in_document + 1
    second_sentence_start = second_sentence[0].index_in_document

    # move the indices of the first sentence
    for i, token in enumerate(first_sentence):
        token_index_mapping[token.index_in_document] = (
            second_sentence_start + i - offset
        )

    # move the indices of the second sentence
    for i, token in enumerate(second_sentence):
        token_index_mapping[token.index_in_document] = first_sentence_start + i

    # finally, correct all token indices between first and second sentence
    for token in document.tokens[first_sentence_end:second_sentence_start]:
        token_index_mapping[token.index_in_document] -= offset

    # sanity checks
    assert len(set(token_index_mapping.values())) == len(
        document.tokens
    ), f"Mapping has duplicates: {token_index_mapping}"
    assert (
        max(token_index_mapping.values()) == len(document.tokens) - 1
    ), f"Mapping has an index out of range ({max(token_index_mapping.values())}): {token_index_mapping}"
    assert (
        min(token_index_mapping.values()) == 0
    ), f"Mapping has an index out of range ({min(token_index_mapping.values())}): {token_index_mapping}"

    # apply the swap now
    tokens = list(document.tokens)
    for i, token in enumerate(tokens):
        document.tokens[token_index_mapping[i]] = PetToken(
            text=token.text,
            pos_tag=token.pos_tag,
            sentence_index=sentence_id_map[token.sentence_index],
            index_in_document=token_index_mapping[token.index_in_document],
        )
    for i, mention in enumerate(document.mentions):
        document.mentions[i] = PetMention(
            type=mention.type,
            token_document_indices=tuple(
                token_index_mapping[i] for i in mention.token_document_indices
            ),
        )


def text_to_tokens(
    document: PetDocument, texts: typing.List[str], index_in_document: int
) -> typing.List[PetToken]:
    text_pos = nltk.pos_tag(texts)
    indices = range(index_in_document, index_in_document + len(texts))
    sentence_id = 0
    if index_in_document > 0:
        sentence_id = document.tokens[index_in_document - 1].sentence_index
    tokens = [
        PetToken(text=t, pos_tag=p, index_in_document=i, sentence_index=sentence_id)
        for t, p, i in zip(texts, text_pos, indices)
    ]
    return tokens


def delete_token_inplace(doc: PetDocument, token_id: int) -> None:
    mention_to_remove = None
    for m_id, m in enumerate(doc.mentions):
        if token_id in m.token_document_indices:
            if len(m.token_document_indices) == 1:
                mention_to_remove = m_id
            else:
                doc.mentions[m_id] = PetMention(
                    token_document_indices=tuple(
                        i for i in m.token_document_indices if i != token_id
                    ),
                    type=m.type,
                )
    if mention_to_remove is not None:
        delete_mention_inplace(doc, mention_to_remove)

    doc.tokens.pop(token_id)
    for i in range(token_id, len(doc.tokens)):
        t = doc.tokens[i]
        doc.tokens[i] = PetToken(
            text=t.text,
            pos_tag=t.pos_tag,
            sentence_index=t.sentence_index,
            index_in_document=t.index_in_document - 1,
        )


def delete_mention_inplace(doc: PetDocument, mention_id: int) -> None:
    # adjust entities
    entity_to_remove = None
    for e_id, e in enumerate(doc.entities):
        if mention_id in e.mention_indices:
            if len(e.mention_indices) == 1:
                entity_to_remove = e_id
            else:
                doc.entities[e_id] = PetEntity(
                    mention_indices=tuple(
                        i for i in e.mention_indices if i != mention_id
                    ),
                )
    if entity_to_remove is not None:
        delete_entity_inplace(doc, entity_to_remove)

    # adjust relations
    relation_to_remove = None
    for r_id, r in enumerate(doc.relations):
        if r.head_mention_index == mention_id or r.tail_mention_index == mention_id:
            relation_to_remove = r_id
    if relation_to_remove is not None:
        delete_relation_inplace(doc, relation_to_remove)

    doc.mentions.pop(mention_id)


def delete_relation_inplace(doc: PetDocument, relation_id: int) -> None:
    doc.relations.pop(relation_id)


def delete_entity_inplace(doc: PetDocument, entity_id: int) -> None:
    doc.entities.pop(entity_id)


def replace_sequence_inplace(
    doc: PetDocument,
    sequence_start: int,
    sequence_end: int,
    replacement: typing.List[str],
) -> None:
    old_sequence_length = sequence_end - sequence_start
    new_sequence_length = len(replacement)
    replacement_tokens = text_to_tokens(doc, replacement, sequence_start)

    if old_sequence_length == new_sequence_length:
        for i, replacement_token in enumerate(replacement_tokens):
            doc.tokens[sequence_start + i] = replacement_token

    if old_sequence_length > new_sequence_length:
        for i, replacement_token in enumerate(replacement_tokens):
            doc.tokens[sequence_start + i] = replacement_token
        for _ in range(old_sequence_length - new_sequence_length):
            delete_token_inplace(doc, sequence_start + new_sequence_length)

    if old_sequence_length < new_sequence_length:
        for i, replacement_token in enumerate(replacement_tokens[:old_sequence_length]):
            doc.tokens[sequence_start + i] = replacement_token
        insert_tokens_inplace(
            doc, replacement_tokens[old_sequence_length:], sequence_end
        )


def replace_mention_text_inplace(
    doc: PetDocument, mention_index: int, new_token_texts: typing.List[str]
) -> None:
    mention = doc.mentions[mention_index]
    replace_sequence_inplace(
        doc,
        mention.token_document_indices[0],
        mention.token_document_indices[-1] + 1,
        new_token_texts,
    )


def get_pos_tag(token_texts: typing.List[str]):
    tagged_text = nltk.pos_tag(token_texts)
    tags = [tagged_text[i][1] for i in range(len(token_texts))]
    return tags
