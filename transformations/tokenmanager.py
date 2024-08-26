import typing

import nltk

from data import PetDocument, PetToken, PetEntity, PetMention


def insert_tokens_inplace(doc: PetDocument, tokens: typing.List[PetToken], index_in_document: int) -> None:
    # first, adjust all mention token indices that are affected
    for m_id, m in enumerate(doc.mentions):
        new_token_ids = []
        affected = False
        # will be true, if the inserted sequence is inside a mention or "touches" it on the right side
        expanded = min(m.token_document_indices) < index_in_document <= max(m.token_document_indices)
        for token_id in m.token_document_indices:
            if token_id < index_in_document:
                new_token_ids.append(token_id)
            else:
                affected = True
                new_token_ids.append(token_id + len(tokens))
        if affected:
            if expanded:
                # inserted tokens into a mention!
                new_token_ids += list(range(index_in_document, index_in_document + len(tokens)))
                new_token_ids = sorted(new_token_ids)
            doc.mentions[m_id] = PetMention(
                type=m.type,
                token_document_indices=tuple(new_token_ids)
            )

    for token in tokens:
        doc.tokens.insert(index_in_document, token)

    # adjust all following tokens
    for i in range(index_in_document + len(tokens), len(doc.tokens)):
        t = doc.tokens[i]
        doc.tokens[i] = PetToken(
            text=t.text,
            pos_tag=t.pos_tag,
            sentence_index=t.sentence_index,
            index_in_document=t.index_in_document + len(tokens)
        )


def delete_token_inplace(doc: PetDocument, token_id: int) -> None:
    mention_to_remove = None
    for m_id, m in enumerate(doc.mentions):
        if token_id in m.token_document_indices:
            if len(m.token_document_indices) == 1:
                mention_to_remove = m_id
            else:
                doc.mentions[m_id] = PetMention(
                    token_document_indices=tuple(i for i in m.token_document_indices if i != token_id),
                    type=m.type,
                )
    if mention_to_remove is not None:
        delete_mention_inplace(doc, mention_to_remove)


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


def delete_relation_inplace(doc: PetDocument, relation_id: int) -> None:
    doc.relations.pop(relation_id)


def delete_entity_inplace(doc: PetDocument, entity_id: int) -> None:
    doc.entities.pop(entity_id)


def replace_sequence_inplace(doc: PetDocument, sequence_start: int, sequence_end: int,
                             replacement: typing.List[str]) -> None:
    old_sequence_length = sequence_end - sequence_start
    new_sequence_length = len(replacement)

    replacement_pos = nltk.pos_tag(replacement)
    replacement_indices = range(sequence_start, sequence_start + len(replacement))
    replacement_sentence_ids = [t.sentence_index for t in doc.tokens[sequence_start:sequence_end]]
    # pad with last sentence index
    replacement_sentence_ids += replacement_sentence_ids[-1] * (new_sequence_length - old_sequence_length)
    replacement_tokens = [
        PetToken(text=t, pos_tag=p, index_in_document=i, sentence_index=s) for t, p, i, s in
        zip(replacement, replacement_pos, replacement_indices, replacement_sentence_ids)
    ]

    if old_sequence_length == new_sequence_length:
        for i in range(sequence_start, sequence_end):
            doc.tokens[i] = replacement_tokens[i]

    if old_sequence_length > new_sequence_length:
        for i in range(sequence_start, sequence_end):
            if i < len(replacement_tokens):
                doc.tokens[i] = replacement_tokens[i]
            else:
                delete_token_inplace(doc, i)

    if old_sequence_length < new_sequence_length:
        for i in range(sequence_start, sequence_end):
            doc.tokens[i] = replacement_tokens[i]
        insert_tokens_inplace(doc, replacement_tokens[sequence_end:], sequence_end - 1)


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
