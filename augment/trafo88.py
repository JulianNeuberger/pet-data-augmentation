import random
import typing

import spacy

import data
from augment import base, params
from data import model


class Trafo88Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
    ):
        super().__init__(dataset)
        self.nlp = spacy.load("en_core_web_sm")

    def do_augment(self, doc: model.Document) -> model.Document:
        return self.sentence_reordering(doc)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @staticmethod
    def _get_new_ordering(document: model.Document) -> typing.List[int]:
        new_ids = list(range(len(document.sentences)))
        random.shuffle(new_ids)
        return new_ids

    @staticmethod
    def sentence_reordering(document: data.Document):
        document = document.copy()
        new_ids = Trafo88Step._get_new_ordering(document)
        id_mapping = {old: new for old, new in enumerate(new_ids)}
        reverse_mapping = {v: k for k, v in id_mapping.items()}

        for mention in document.mentions:
            mention.sentence_index = id_mapping[mention.sentence_index]
        for relation in document.relations:
            relation.evidence = [id_mapping[e] for e in relation.evidence]
        for sentence in document.sentences:
            for token in sentence.tokens:
                token.sentence_index = id_mapping[token.sentence_index]
        document.sentences = [
            document.sentences[reverse_mapping[i]]
            for i in range(len(document.sentences))
        ]
        return document
