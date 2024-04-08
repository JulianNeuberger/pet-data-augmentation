import mockito

from augment import trafo88
from augment.trafo86 import Trafo86HypernymReplacement, Trafo86HyponymReplacement
from data import model


def test_do_augment():
    document = model.Document(
        text="First test. This is a longer sentence.",
        name="test",
        sentences=[
            model.Sentence(tokens=[
                model.Token(text="First", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0),
                model.Token(text="test", index_in_document=1, pos_tag="", bio_tag="", sentence_index=0),
                model.Token(text=".", index_in_document=2, pos_tag="", bio_tag="", sentence_index=0),
            ]),
            model.Sentence(tokens=[
                model.Token(text="This", index_in_document=3, pos_tag="", bio_tag="", sentence_index=1),
                model.Token(text="is", index_in_document=4, pos_tag="", bio_tag="", sentence_index=1),
                model.Token(text="a", index_in_document=5, pos_tag="", bio_tag="", sentence_index=1),
                model.Token(text="longer", index_in_document=6, pos_tag="", bio_tag="", sentence_index=1),
                model.Token(text="sentence", index_in_document=7, pos_tag="", bio_tag="", sentence_index=1),
                model.Token(text=".", index_in_document=8, pos_tag="", bio_tag="", sentence_index=1),
            ]),
            model.Sentence(tokens=[
                model.Token(text="And", index_in_document=9, pos_tag="", bio_tag="", sentence_index=2),
                model.Token(text="a", index_in_document=10, pos_tag="", bio_tag="", sentence_index=2),
                model.Token(text="final", index_in_document=11, pos_tag="", bio_tag="", sentence_index=2),
                model.Token(text="one", index_in_document=12, pos_tag="", bio_tag="", sentence_index=2),
                model.Token(text=".", index_in_document=13, pos_tag="", bio_tag="", sentence_index=2),
            ]),
        ],
        mentions=[
            model.Mention(ner_tag="", sentence_index=1, token_indices=[3, 4]),
            model.Mention(ner_tag="", sentence_index=0, token_indices=[1])
        ],
        relations=[
            model.Relation(0, 1, "", [1])
        ]
    )

    step = trafo88.Trafo88Step([document])

    mockito.when(trafo88.Trafo88Step)._get_new_ordering(mockito.any(model.Document)).thenReturn([2, 0, 1])

    assert trafo88.Trafo88Step._get_new_ordering(document) == [2, 0, 1]

    augmented = step.do_augment(document)

    assert augmented != step
    assert augmented.sentences[0].text == "This is a longer sentence ."
    assert augmented.mentions[0].sentence_index == 0
    assert augmented.mentions[0].token_indices == [3, 4]
