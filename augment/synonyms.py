import random
import typing

import nltk
from nltk.corpus import stopwords, wordnet

import data
from augment import base, params
from data import PetDocument, PetToken
from pos_enum import Pos

nltk.download("stopwords")


class SynonymInsertion(base.BaseTokenReplacementStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/synonym_insertion
    B.100
    """

    def __init__(
        self, dataset: typing.List[PetDocument], replace_probability: float, **kwargs
    ):
        super().__init__(dataset, replace_probability, **kwargs)
        self.seed = 42
        self.stopwords = stopwords.words("english")
        random.seed(self.seed)
        self.relevant_pos = Pos.VERB.tags + Pos.AD.tags + Pos.NOUN.tags

    @staticmethod
    def get_default_configuration(
        dataset: typing.List[PetDocument],
    ) -> "SynonymInsertion":
        return SynonymInsertion(dataset, replace_probability=0.15)

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

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = []

        for token in doc.tokens:
            if token.text in self.stopwords:
                continue
            if token.pos_tag not in self.relevant_pos:
                continue

            text = token.text
            synsets = wordnet.synsets(text, pos=self.get_wordnet_pos(token.pos_tag))
            synsets = [s.name().split(".")[0] for s in synsets]
            synsets = [s for s in synsets if s.lower() != text]
            synsets = list(set(synsets))
            if len(synsets) == 0:
                continue

            candidates.append([token])

        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        assert len(candidate) == 1
        token = candidate[0]
        text = candidate[0].text
        synsets = wordnet.synsets(text, pos=self.get_wordnet_pos(token.pos_tag))
        synsets = [s.name().split(".")[0] for s in synsets]
        synsets = [s for s in synsets if s.lower() != text]
        synsets = list(set(synsets))
        random.shuffle(synsets)
        synonyms = [s.replace("_", " ") for s in synsets]
        synonyms = [nltk.tokenize.word_tokenize(s) for s in synonyms]

        replacements = [[text] + s for s in synonyms]

        return replacements[:num_replacements_per_candidate]


class SynonymSubstitution(base.BaseTokenReplacementStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/synonym_substitution
    B.101
    """

    def __init__(
        self, dataset: typing.List[PetDocument], replace_probability: float, **kwargs
    ):
        super().__init__(dataset, replace_probability, **kwargs)
        self.relevant_pos = Pos.VERB.tags + Pos.AD.tags + Pos.NOUN.tags

    @staticmethod
    def get_default_configuration(
        dataset: typing.List[PetDocument],
    ) -> "SynonymSubstitution":
        return SynonymSubstitution(dataset, replace_probability=0.30)

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

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = []

        for token in doc.tokens:
            if token.pos_tag not in self.relevant_pos:
                continue

            text = token.text
            synsets = wordnet.synsets(text, pos=self.get_wordnet_pos(token.pos_tag))
            synsets = [s.name().split(".")[0] for s in synsets]
            synsets = [s for s in synsets if s.lower() != text]
            synsets = list(set(synsets))
            if len(synsets) == 0:
                continue

            candidates.append([token])

        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        assert len(candidate) == 1
        token = candidate[0]
        text = candidate[0].text
        synsets = wordnet.synsets(text, pos=self.get_wordnet_pos(token.pos_tag))
        synsets = [s.name().split(".")[0] for s in synsets]
        synsets = [s for s in synsets if s.lower() != text]
        synsets = list(set(synsets))
        random.shuffle(synsets)
        synonyms = [s.replace("_", " ") for s in synsets]
        synonyms = [nltk.tokenize.word_tokenize(s) for s in synonyms]

        return synonyms[:num_replacements_per_candidate]


if __name__ == "__main__":

    def main():
        doc = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()[0]
        step = SynonymInsertion([], 0.5)
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
