import inspect
import typing

from augment import base
from data import PetDocument, PetToken, PetRelation, PetEntity, PetMention


def document_fixture():
    doc = PetDocument(
        id="",
        text="",
        name="test",
        category="",
        tokens=[
            # sentence 1
            PetToken("This", 0, "A", 0),
            PetToken("is", 1, "B", 0),
            PetToken("a", 2, "C", 0),
            PetToken("sentence", 3, "A", 0),
            PetToken(".", 4, "A", 0),
            # sentence 2
            PetToken("And", 5, "A", 1),
            PetToken("another", 6, "B", 1),
            PetToken("one", 7, "C", 1),
            PetToken("!", 8, "A", 1),
            # sentence 3
            PetToken("Here", 9, "A", 2),
            PetToken("comes", 10, "B", 2),
            PetToken("the", 11, "C", 2),
            PetToken("last", 12, "C", 2),
            PetToken("one", 13, "C", 2),
            PetToken(".", 14, "A", 2),
        ],
        mentions=[
            PetMention("Object", tuple([3])),
            PetMention("Object", tuple([1, 2])),
            PetMention("Something", tuple([10])),
            PetMention("Activity", tuple([12])),
            PetMention("Object", tuple([13])),
        ],
        entities=[
            PetEntity(tuple([0, 1])),
            PetEntity(tuple([2])),
            PetEntity(tuple([3])),
            PetEntity(tuple([4])),
        ],
        relations=[
            PetRelation("Testtag", 0, 1),
            PetRelation("Othertag", 1, 4),
            PetRelation("Lasttag", 3, 1),
        ],
    )
    return doc
