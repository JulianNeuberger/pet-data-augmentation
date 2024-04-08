import random
import typing

import nltk
from nltk.corpus import stopwords, wordnet

from augment import base, params
from data import model
from pos_enum import Pos

nltk.download("stopwords")


class Trafo100Step(base.BaseTokenReplacementStep):
    def __init__(self, dataset: typing.List[model.Document], prob=0.5, n: int = 10):
        super().__init__(dataset, n)
        self.seed = 42
        self.prob = prob
        self.stopwords = stopwords.words("english")
        random.seed(self.seed)
        self.relevant_pos = Pos.VERB.tags + Pos.AD.tags + Pos.NOUN.tags

    @staticmethod
    def get_wordnet_pos(treebank_tag: str):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return ""

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="prob", min_value=0.0, max_value=1.0),
        ]

    def get_replacement_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        candidates = []

        for token in doc.tokens:
            if token.text in self.stopwords:
                continue
            if token.pos_tag not in self.relevant_pos:
                continue
            candidates.append([token])

        return candidates

    def get_replacement(
        self, candidate: typing.List[model.Token]
    ) -> typing.Optional[typing.List[str]]:
        assert len(candidate) == 1
        token = candidate[0]
        text = candidate[0].text
        synsets = wordnet.synsets(text, pos=self.get_wordnet_pos(token.pos_tag))
        synsets = [s.name().split(".")[0] for s in synsets]
        synsets = [s for s in synsets if s.lower() != text]
        synsets = list(set(synsets))
        if len(synsets) == 0:
            return None
        if random.random() >= self.prob:
            return None
        synonym = random.choice(synsets)
        synonym = synonym.replace("_", " ")
        return synonym.split()
