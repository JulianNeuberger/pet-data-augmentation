import copy
import typing
import random

import spacy

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo6Step(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document]):
        super().__init__(dataset)
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    def do_augment(self, doc: model.Document, num_augments: int) -> typing.List[model.Document]:
        doc = doc.copy()
        candidates: typing.List[model.Token] = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.text.lower() == "not":
                    candidates.append(token)

                spacy_document = self.nlp(token.text)
                if len(spacy_document) <= 1:
                    continue

                if spacy_document[1].text == "n't":
                    candidates.append(token)

        random.shuffle(candidates)
        for token in candidates:
            if token.text.lower() == "not":
                tokenmanager.delete_token(doc, token.index_in_document)
                continue

            spacy_document = self.nlp(token.text)

            if spacy_document[1].text == "n't":
                token.text = spacy_document[0].text

        return [doc]
