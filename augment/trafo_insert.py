import random
import typing

from augment import base, params
from data import model
from transformations import tokenmanager


class TrafoInsertStep(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], count_insertions=1):
        super().__init__(dataset)
        self.count_insertions = count_insertions
        vocab = set()
        for document in dataset:
            for token in document.tokens:
                vocab.add(token.text)
        self.vocab = list(vocab)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="count_insertions", max_value=20, min_value=1),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        for _ in range(self.count_insertions):
            index_in_doc = random.randint(0, len(doc.tokens))
            token_text = random.choice(self.vocab)

            tokenmanager.insert_token_text_into_document(
                doc=doc, token_text=token_text, index_in_document=index_in_doc
            )
        return doc
