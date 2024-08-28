import typing

from data import PetDocument, PetMention, PetToken


def decode_predictions(
    document: PetDocument, predictions: typing.List[typing.List[str]]
) -> PetDocument:
    print(predictions)
    assert len(document.sentences) == len(predictions)

    tokens = []
    mentions = []

    for sent_id, (sentence, predicted_tags) in enumerate(
        zip(document.sentences, predictions)
    ):
        current_mention_tag: typing.Optional[str] = None
        current_mention_token_indices: typing.List[int] = []
        for token, bio_tag in zip(sentence, predicted_tags):
            current_token = PetToken(
                text=token.text,
                pos_tag=token.pos_tag,
                index_in_document=token.index_in_document,
                sentence_index=sent_id,
            )
            tokens.append(current_token)

            bio_tag = bio_tag.strip()
            tag = bio_tag.split("-", 1)[-1]

            is_entity_start = bio_tag.startswith("B-")

            should_finish_entity = is_entity_start or tag == "O"

            if should_finish_entity and current_mention_tag is not None:
                mentions.append(
                    PetMention(
                        type=current_mention_tag,
                        token_document_indices=tuple(current_mention_token_indices),
                    )
                )
                current_mention_tag = None

            if is_entity_start:
                current_mention_tag = tag

            if current_mention_tag is not None:
                current_mention_token_indices.append(current_token.index_in_document)

        if current_mention_tag is not None:
            mentions.append(
                PetMention(
                    type=current_mention_tag,
                    token_document_indices=tuple(current_mention_token_indices),
                )
            )

    return PetDocument(
        name=document.name,
        text=document.text,
        category=document.category,
        id=document.id,
        tokens=tokens,
        mentions=mentions,
        relations=[],
        entities=[],
    )
