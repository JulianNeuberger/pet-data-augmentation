from data import PetDocument, PetToken


def test_tokens_for_character_indices():
    tokens = [
        PetToken("This", index_in_document=0, sentence_index=0, pos_tag=""),
        PetToken("is", index_in_document=1, sentence_index=0, pos_tag=""),
        PetToken("a", index_in_document=2, sentence_index=0, pos_tag=""),
        PetToken("test", index_in_document=3, sentence_index=0, pos_tag=""),
        PetToken(".", index_in_document=4, sentence_index=0, pos_tag=""),
    ]

    doc = PetDocument(
        id="",
        name="",
        category="",
        text="",
        tokens=tokens,
        entities=[],
        relations=[],
        mentions=[],
    )

    found_tokens = doc.tokens_for_character_indices(0, 4)
    assert len(found_tokens) == 1
    assert found_tokens[0].text == "This"

    found_tokens = doc.tokens_for_character_indices(4, 0)
    assert len(found_tokens) == 0

    found_tokens = doc.tokens_for_character_indices(0, 7)
    assert len(found_tokens) == 2
    assert found_tokens[0].text == "This"
    assert found_tokens[1].text == "is"

    found_tokens = doc.tokens_for_character_indices(4, 8)
    assert len(found_tokens) == 1
    assert found_tokens[0].text == "is"
