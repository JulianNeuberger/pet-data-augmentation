import typing

import nltk

from data import PetDocument, PetToken, PetEntity, PetMention, PetRelation


def insert_tokens_inplace(
    doc: PetDocument,
    tokens: typing.List[PetToken],
    index_in_document: int,
    expand_mention_id: typing.Optional[int] = None,
) -> None:
    if expand_mention_id is not None:
        expanded_mention = doc.mentions[expand_mention_id]
        assert (
            min(expanded_mention.token_document_indices)
            <= index_in_document
            <= max(expanded_mention.token_document_indices) + 1
        ), (
            f"Trying to insert tokens at {index_in_document}, "
            f"but this index is not inside or adjacent to the "
            f"mention (id={expand_mention_id}, "
            f"start={min(expanded_mention.token_document_indices)}, "
            f"end={max(expanded_mention.token_document_indices)}) "
            f"that should be expanded by it."
        )

    # first, adjust all mention token indices that are affected
    for m_id, m in enumerate(doc.mentions):
        new_token_ids = []
        affected = False
        # will be true, if the inserted sequence is inside a mention
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
        if affected or m_id == expand_mention_id:
            if expanded or m_id == expand_mention_id:
                # inserted tokens into a mention!
                expanded_token_ids = list(
                    range(index_in_document, index_in_document + len(tokens))
                )
                new_token_ids += expanded_token_ids
                new_token_ids = sorted(new_token_ids)
            assert len(new_token_ids) != 0
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
    text_pos = [p for _, p in nltk.pos_tag(texts)]
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
    mention_id_to_remove = None
    for m_id, m in enumerate(doc.mentions):
        new_token_indices = []
        for i in m.token_document_indices:
            if i == token_id:
                continue
            index = i
            if index > token_id:
                index -= 1
            new_token_indices.append(index)
        if len(new_token_indices) == 0:
            mention_id_to_remove = m_id
        else:
            doc.mentions[m_id] = PetMention(
                token_document_indices=tuple(new_token_indices),
                type=m.type,
            )
    if mention_id_to_remove is not None:
        delete_mention_inplace(doc, mention_id_to_remove)

    token = doc.tokens[token_id]

    sentence_adjustment = 0
    sentence = doc.sentences[token.sentence_index]
    if len(sentence) == 1:
        # final token in this sentence
        sentence_adjustment = 1

    doc.tokens.pop(token_id)
    for i in range(token_id, len(doc.tokens)):
        t = doc.tokens[i]
        doc.tokens[i] = PetToken(
            text=t.text,
            pos_tag=t.pos_tag,
            sentence_index=t.sentence_index - sentence_adjustment,
            index_in_document=t.index_in_document - 1,
        )


def delete_mention_inplace(doc: PetDocument, mention_id: int) -> None:
    assert all(
        [r.head_mention_index < len(doc.mentions) for r in doc.relations]
    ), "broken before"
    assert all(
        [r.tail_mention_index < len(doc.mentions) for r in doc.relations]
    ), "broken before"

    # adjust entities
    entity_to_remove = None
    for e_id, e in enumerate(doc.entities):
        new_mention_ids = []
        for i in e.mention_indices:
            if i == mention_id:
                continue
            if i > mention_id:
                i -= 1
            new_mention_ids.append(i)

        if len(new_mention_ids) == 0:
            entity_to_remove = e_id
        else:
            doc.entities[e_id] = PetEntity(
                mention_indices=tuple(new_mention_ids),
            )
    if entity_to_remove is not None:
        delete_entity_inplace(doc, entity_to_remove)

    # adjust relations
    relations_to_remove = []
    for r_id, r in enumerate(doc.relations):
        if r.head_mention_index == mention_id:
            relations_to_remove.append(r_id)
            continue

        if r.tail_mention_index == mention_id:
            relations_to_remove.append(r_id)
            continue

        new_head_id = r.head_mention_index
        new_tail_id = r.tail_mention_index
        if r.head_mention_index > mention_id:
            new_head_id -= 1
        if r.tail_mention_index > mention_id:
            new_tail_id -= 1

        assert new_head_id < len(doc.mentions) - 1
        assert new_tail_id < len(doc.mentions) - 1

        doc.relations[r_id] = PetRelation(
            type=r.type, head_mention_index=new_head_id, tail_mention_index=new_tail_id
        )

    for relation_to_remove in sorted(relations_to_remove, reverse=True):
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
    affected_mention = doc.get_mention_for_token(doc.tokens[sequence_start])
    affected_mention_id = None
    if affected_mention:
        # make sure we affect parts of entities (but no non-entity text),
        # or the whole entity, but never parts of the entity and non-entity text
        replaced_text = " ".join(
            [doc.tokens[i].text for i in range(sequence_start, sequence_end)]
        )
        replacement_text = " ".join(replacement)
        assert min(affected_mention.token_document_indices) <= sequence_start, (
            f"Trying to replace '{replaced_text}' ({sequence_start}-{sequence_end}) "
            f"with '{replacement_text}', but sequence start ({sequence_start}) is "
            f"smaller than the affected mention with text '{affected_mention.text(doc)}' "
            f"(tokens {affected_mention.token_document_indices})"
        )
        assert max(affected_mention.token_document_indices) >= sequence_end - 1, (
            f"Trying to replace '{replaced_text}' ({sequence_start}-{sequence_end}) "
            f"with '{replacement_text}', but sequence end ({sequence_end}) is bigger "
            f"than the affected mention with text '{affected_mention.text(doc)}' "
            f"(tokens {affected_mention.token_document_indices})"
        )
        affected_mention_id = doc.mentions.index(affected_mention)

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
            doc,
            replacement_tokens[old_sequence_length:],
            sequence_end,
            expand_mention_id=affected_mention_id,
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
