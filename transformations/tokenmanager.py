import typing

import nltk

from data import model


def delete_token(doc: model.Document, index_in_document: int):
    """delete token from sentence.tokens, deleting corresponding
    mention, if it was its last token.
    """
    sentence_index, index_in_sentence = delete_token_from_tokens(doc, index_in_document)
    mention_index = delete_token_from_mention_token_indices(
        doc, index_in_sentence, sentence_index
    )


def delete_token_from_tokens(
    doc: model.Document, index_in_document: int
) -> typing.Optional[typing.Tuple[int, int]]:
    token = doc.tokens[index_in_document]
    sentence_id = token.sentence_index
    index_in_sentence = -1

    assert sentence_id < len(doc.sentences), (
        f"Got sentence index of {sentence_id}, "
        f"but doc only has {len(doc.sentences)} sentences, "
        f"sentence ids of tokens are {list(set([t.sentence_index for t in doc.tokens]))}"
    )

    for i, token in enumerate(doc.sentences[sentence_id].tokens):
        if token.index_in_document == index_in_document:
            index_in_sentence = i
            break

    doc.sentences[sentence_id].tokens.pop(index_in_sentence)

    for subsequent_token in doc.tokens[index_in_document:]:
        subsequent_token.index_in_document -= 1

    return sentence_id, index_in_sentence


# Author: Benedikt
def delete_token_from_mention_token_indices(
    doc: model.Document, token_index_in_sentence: int, sentence_index: int
):
    mention_to_delete = None
    for mention_id, mention in enumerate(doc.mentions):
        if mention.sentence_index != sentence_index:
            continue

        if token_index_in_sentence in mention.token_indices:
            if len(mention.token_indices) == 1:
                doc.mentions.remove(mention)
                delete_mention_from_entities(doc, mention_id)
                adjust_mention_indices_in_entities(doc, mention_id)
                mention_to_delete = mention_id
            else:
                mention.token_indices.remove(token_index_in_sentence)
                for i in range(len(mention.token_indices)):
                    if mention.token_indices[i] > token_index_in_sentence:
                        mention.token_indices[i] -= 1
        else:
            for i in range(len(mention.token_indices)):
                if mention.token_indices[i] > token_index_in_sentence:
                    mention.token_indices[i] -= 1
        mention_id += 1
    return mention_to_delete


def adjust_mention_indices_in_entities(doc: model.Document, mention_index: int):
    for entity in doc.entities:
        for i in range(len(entity.mention_indices)):
            if entity.mention_indices[i] > mention_index:
                entity.mention_indices[i] -= 1


def adjust_entity_indices_in_relations(doc: model.Document, entity_index: int):
    for relation in doc.relations:
        if relation.head_entity_index > entity_index:
            relation.head_entity_index -= 1

        if relation.tail_entity_index > entity_index:
            relation.tail_entity_index -= 1


def delete_mention_from_entities(doc: model.Document, mention_index: int):
    for entity_id, entity in enumerate(doc.entities):
        if mention_index not in entity.mention_indices:
            continue

        is_last_mention = len(entity.mention_indices) == 1

        if is_last_mention:
            doc.entities.remove(entity)
            delete_relations(doc, entity_id)
            adjust_entity_indices_in_relations(doc, entity_id)
            break
        else:
            entity.mention_indices.remove(mention_index)


def delete_relations(doc: model.Document, entity_index: int):
    affected_relation_indices = set()
    for relation_index, relation in enumerate(doc.relations):
        if relation.head_entity_index == entity_index:
            affected_relation_indices.add(relation_index)
        if relation.tail_entity_index == entity_index:
            affected_relation_indices.add(relation_index)
    for relation_index in sorted(list(affected_relation_indices), reverse=True):
        doc.relations.pop(relation_index)


def create_token(
    doc: model.Document,
    token: model.Token,
    index_in_sentence: int,
    mention_index=None,
):
    if mention_index is None:
        insert_token_in_tokens(doc, token, index_in_sentence)
        return

    # a new given Token gets inserted in the token array of the belonging sentence and all index_in_doc are adjusted
    insert_token_in_tokens(doc, token, index_in_sentence)
    # adds token_id to mention if a mention id is given
    insert_token_in_mentions(doc, index_in_sentence, mention_index)


def expand_token(doc: model.Document, token: model.Token, new_tokens: typing.List[str]):
    assert len(new_tokens) > 0

    token.text = new_tokens[0]
    token_index_in_sentence = token.index_in_sentence(doc)

    for i, remaining_token in enumerate(new_tokens[1:]):
        mention_indices = get_mentions(
            doc, token_index_in_sentence, token.sentence_index
        )
        mention_index = mention_indices[0] if len(mention_indices) > 0 else None
        new_token = model.Token(
            text=remaining_token,
            index_in_document=token.index_in_document + i,
            pos_tag=get_pos_tag([remaining_token])[0],
            bio_tag=get_continued_bio_tag(token.bio_tag),
            sentence_index=token.sentence_index,
        )
        create_token(doc, new_token, token_index_in_sentence + i, mention_index)


def replace_sequence_text_in_sentence(
    doc: model.Document,
    sentence_id: int,
    start_in_sentence: int,
    stop_in_sentence: int,
    new_token_texts: typing.List[str],
):
    offset = doc.token_offset_for_sentence(sentence_id)
    start_in_doc = offset + start_in_sentence
    stop_in_doc = offset + stop_in_sentence

    old_tokens = doc.tokens[start_in_doc:stop_in_doc]

    old_bio_tag = get_bio_tag_short(doc.tokens[start_in_doc].bio_tag)
    if old_bio_tag == "O":
        new_bio_tags = ["O"] * len(new_token_texts)
    else:
        new_bio_tags = [f"B-{old_bio_tag}"]
        new_bio_tags += [f"I-{old_bio_tag}"] * (len(new_token_texts) - 1)

    new_pos_tags = get_pos_tag(new_token_texts)

    new_tokens = [
        model.Token(
            text=new_text,
            pos_tag=new_pos,
            index_in_document=start_in_doc + new_index,
            bio_tag=new_bio_tag,
            sentence_index=sentence_id,
        )
        for new_text, new_pos, new_index, new_bio_tag in zip(
            new_token_texts, new_pos_tags, range(len(new_token_texts)), new_bio_tags
        )
    ]

    length_difference = len(new_tokens) - len(old_tokens)

    if length_difference == 0:
        for i, token in enumerate(new_tokens):
            doc.sentences[sentence_id].tokens[start_in_sentence + i] = token
        return

    if length_difference < 0:
        for old_token in old_tokens[len(new_tokens) :]:
            delete_token(doc, old_token.index_in_document)
        for i, token in enumerate(new_tokens):
            doc.sentences[sentence_id].tokens[start_in_sentence + i] = token

    if length_difference > 0:
        for i, token in enumerate(new_tokens[: len(old_tokens)]):
            doc.sentences[sentence_id].tokens[start_in_sentence + i] = token
        for i, new_token in enumerate(new_tokens[len(old_tokens) :]):
            mentions = doc.get_mentions_for_token(old_tokens[0])
            mention_id = None
            if len(mentions) > 0:
                mention_id = doc.mention_index(mentions[0])

            create_token(
                doc,
                new_token,
                start_in_sentence + len(old_tokens) + i,
                mention_id,
            )


def replace_mention_text(
    doc: model.Document, mention_index: int, new_token_texts: typing.List[str]
):
    mention = doc.mentions[mention_index]
    replace_sequence_text_in_sentence(
        doc,
        mention.sentence_index,
        mention.token_indices[0],
        mention.token_indices[-1] + 1,
        new_token_texts,
    )


def insert_token_in_tokens(
    doc: model.Document, token: model.Token, index_in_sentence: int
):
    sentence_index = token.sentence_index
    doc.sentences[sentence_index].tokens.insert(index_in_sentence, token)
    for j in range(
        index_in_sentence + 1, len(doc.sentences[sentence_index].tokens)
    ):
        doc.sentences[sentence_index].tokens[j].index_in_document += 1
    for i in range(sentence_index + 1, len(doc.sentences)):
        for tok in doc.sentences[i].tokens:
            tok.index_in_document += 1
    for mention in doc.mentions:
        if mention.sentence_index == sentence_index:
            for i in range(len(mention.token_indices)):
                if mention.token_indices[i] >= index_in_sentence:
                    mention.token_indices[i] += 1


def insert_token_in_mentions(
    doc: model.Document, index_in_sentence: int, mention_id: int
):
    if index_in_sentence not in doc.mentions[mention_id].token_indices:
        doc.mentions[mention_id].token_indices.append(index_in_sentence)
        doc.mentions[mention_id].token_indices.sort()


def insert_token_text_into_document(
    doc: model.Document, token_text: str, index_in_document: int
) -> None:
    """
    Inserts new text as a single token, expanding mentions if it is
    inserted right after a mention span.
    """
    bio_tag = "O"
    sentence_id = 0
    index_in_sentence = 0
    mention_id: typing.Optional[int] = None
    if index_in_document > 0:
        previous_token = doc.tokens[index_in_document - 1]
        bio_tag = get_continued_bio_tag(previous_token.bio_tag)
        sentence_id = previous_token.sentence_index
        index_in_sentence = previous_token.index_in_sentence(doc) + 1
        mentions = doc.get_mentions_for_token(previous_token)
        if len(mentions) > 0:
            mention = mentions[0]
            mention_id = doc.mention_index(mention)

    pos_tags = get_pos_tag([token_text])
    assert len(pos_tags) == 1
    pos_tag = pos_tags[0]

    token = model.Token(
        text=token_text,
        index_in_document=index_in_document,
        pos_tag=pos_tag,
        bio_tag=bio_tag,
        sentence_index=sentence_id,
    )

    insert_token_in_tokens(doc, token, index_in_sentence)
    if mention_id is not None:
        insert_token_in_mentions(doc, index_in_sentence, mention_id)


# Author: Benedikt
# delete a sentence from a model.Document, used for the filters
def delete_sentence(doc: model.Document, sent_index: int):
    # delete every token of the sentence
    sentence = doc.sentences[sent_index]
    first_index_in_doc = sentence.tokens[0].index_in_document
    for i in range(len(sentence.tokens)):
        delete_token(doc, first_index_in_doc)

    # delete the sentence from the model.Document.sentences
    del doc.sentences[sent_index]

    # change the sentence_index of every token of the following sentences
    for i in range(sent_index, len(doc.sentences)):
        for token in doc.sentences[i].tokens:
            token.sentence_index -= 1

    # change the sentence index of every mention with sentence_index > sent_index
    for mention in doc.mentions:
        if mention.sentence_index > sent_index:
            mention.sentence_index -= 1


def get_pos_tag(token_texts: typing.List[str]):
    tagged_text = nltk.pos_tag(token_texts)
    tags = [tagged_text[i][1] for i in range(len(token_texts))]
    return tags


def get_continued_bio_tag(previous_tag: str):
    """
    Returns the continuation of the given previous tag.
    E.g., I-Actor if previous tag was either I-Actor or
    B-Actor
    """
    if previous_tag == "O":
        return "O"
    bio = get_bio_tag_short(previous_tag)
    return f"I-{bio}"


# Author: Benedikt
# returns the Bio Tag without the B or I in front
def get_bio_tag_short(tag: str):
    bio = tag.split("-")
    if len(bio) > 1:
        bio_tag = bio[1]
    else:
        bio_tag = bio[0]
    return bio_tag


# Author: Leonie
# returns the index in a sentence of a token, if the token is found in the sentence, otherwise returns None
def get_index_in_sentence(
    sent: model.Sentence, text: typing.List, index_in_doc: int = None
):
    word_count = 0
    ind_in_sent = None
    if sent == [] or text == []:
        return None
    for i in range(len(sent.tokens)):
        if index_in_doc == None:
            if sent.tokens[i].text == text[word_count]:
                word_count += 1
                if word_count == 1:
                    ind_in_sent = i
                if word_count == len(text):
                    break
            else:
                word_count = 0
        elif (
            sent.tokens[i].text == text[word_count]
            and sent.tokens[i].index_in_document == index_in_doc + word_count
        ):
            word_count += 1
            if word_count == 1:
                ind_in_sent = i
            if word_count == len(text):
                break
        else:
            word_count = 0
    if word_count == len(text):  # and ind_in_sent != None:
        return ind_in_sent
    else:
        return None


def get_mentions(doc: model.Document, ind_in_sent, sentence_idx):
    mentions = []
    for i, mention in enumerate(doc.mentions):
        if mention.sentence_index == sentence_idx:
            for j in mention.token_indices:
                if j == ind_in_sent:
                    mentions.append(i)
                break
    return mentions
