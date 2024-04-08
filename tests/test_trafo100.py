from augment import trafo100
from data import model


def test_do_augment():
    tokens = [model.Token(text="good", index_in_document=0,
                          pos_tag="JJ", bio_tag="O",
                          sentence_index=0),
              model.Token(text="leave", index_in_document=1,
                          pos_tag="VBP", bio_tag="O",
                          sentence_index=0),
              model.Token(text="head", index_in_document=2,
                          pos_tag="NN", bio_tag="O",
                          sentence_index=0),
              model.Token(text=".", index_in_document=3,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0)]
    sentence1 = model.Sentence(tokens=tokens)

    doc = model.Document(
        text="good leave head .",
        name="1", sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[])

    trafo = trafo100.Trafo100Step([doc])
    aug = trafo.do_augment(doc)

    print()
    print(" ".join(t.text for t in doc.tokens))
    print(" ".join(t.text for t in aug.tokens))
