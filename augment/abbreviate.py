import abc
import os.path
import pathlib
import typing

import nltk

import data
from augment import params, base, grammaire
from augment.base import BaseTokenReplacementStep
from data import PetDocument, PetToken


class BaseAbbreviationStep(BaseTokenReplacementStep, abc.ABC):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/contraction_expansions
    B.27
    """

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        abbreviations: typing.Dict[str, typing.List[str]],
        case_sensitive: bool = False,
    ):
        super().__init__(dataset, replacements_per_document=5)
        self.case_sensitive = case_sensitive
        self.expansions = abbreviations

        self.contractions: typing.Dict[str, typing.List[str]] = {}
        for key, values in abbreviations.items():
            for value in values:
                if value not in self.contractions:
                    self.contractions[value] = []
                self.contractions[value].append(key)

    def get_contraction_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        return self._get_candidates(self.contractions, doc)

    def get_expansion_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        return self._get_candidates(self.expansions, doc)

    @staticmethod
    def _get_candidates(
        dictionary: typing.Dict[str, typing.List[str]], doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = []
        candidate: typing.List[PetToken] = []
        for token in doc.tokens:
            candidate += [token]
            candidate_key = " ".join(t.text for t in candidate)
            if candidate_key in dictionary:
                candidates.append(candidate)
                candidate = []
                continue
            if BaseAbbreviationStep.has_keys_starting_with(dictionary, candidate_key):
                continue
            candidate = []
        return candidates

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        return self.get_expansion_candidates(doc) + self.get_contraction_candidates(doc)

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        key = " ".join(t.text for t in candidate)
        if key in self.contractions:
            replace_texts = self.contractions[key]
        elif key in self.expansions:
            replace_texts = self.expansions[key]
        else:
            raise AssertionError("Should not happen?")
        assert key not in replace_texts
        return [nltk.tokenize.word_tokenize(text) for text in replace_texts]

    @staticmethod
    def has_keys_starting_with(
        dictionary: typing.Dict[str, typing.Any], partial_key: str
    ) -> bool:
        for key in dictionary.keys():
            if key.startswith(partial_key):
                return True
        return False


class ContractionsAndExpansionsPerturbation(BaseAbbreviationStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/contraction_expansions
    B.27
    """

    def __init__(self, dataset: typing.List[PetDocument]):
        abbreviations = self._load()
        super().__init__(dataset, abbreviations)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @staticmethod
    def _load() -> typing.Dict[str, typing.List[str]]:
        abbreviations = {}
        resources_path = (
            pathlib.Path(__file__).parent.parent.joinpath("resources").resolve()
        )
        lookup_table_path = os.path.join(resources_path, "abbreviations", "27.txt")
        with open(lookup_table_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                key, value = line.split(":", 1)
                abbreviations[key] = [value]
        return abbreviations


class EnglishAndFrenchAbbreviations(base.BaseTokenReplacementStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/insert_abbreviation
    B.52
    """

    def __init__(self, dataset: typing.List[PetDocument]):
        super().__init__(dataset, replacements_per_document=5)
        rules_path = (
            pathlib.Path(__file__)
            .parent.parent.joinpath("resources")
            .joinpath("abbreviations")
            .joinpath("52.txt")
            .resolve()
        )
        with open(rules_path, "r", encoding="utf-8") as f:
            self.grammar = grammaire.compile(f.read())

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        document_text = " ".join(t.text for t in doc.tokens)
        results = grammaire.parse(document_text, self.grammar)

        self.replacements = {}
        candidates = []

        label: str
        start: int
        stop: int
        for label, (start, stop) in results:
            tokens = doc.tokens_for_character_indices(start, stop)
            print(
                f"Parser found label \"{label}\", we retrieved tokens \"{' '.join(t.text for t in tokens)}\""
            )
            candidates.append(tokens)
        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        document_text = " ".join(t.text for t in document.tokens)
        results = grammaire.parse(document_text, self.grammar)
        for label, (start, stop) in results:
            tokens = document.tokens_for_character_indices(start, stop)
            if " ".join(t.text for t in tokens) == " ".join(t.text for t in candidate):
                return [nltk.tokenize.word_tokenize(label)]
        return []

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []


class ReplaceAbbreviationsAndAcronyms(BaseAbbreviationStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/replace_abbreviation_and_acronyms
    B.82
    """

    def __init__(self, dataset: typing.List[PetDocument]):
        abbreviations = self._load()
        super().__init__(dataset, abbreviations)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @staticmethod
    def _load() -> typing.Dict[str, typing.List[str]]:
        abbreviations = {}

        abbreviations_path = (
            pathlib.Path(__file__)
            .parent.parent.joinpath("resources")
            .joinpath("abbreviations")
            .joinpath("82.txt")
            .resolve()
        )
        with open(abbreviations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                key, value = line.split(":", 1)
                if key not in abbreviations:
                    abbreviations[key] = []
                abbreviations[key].append(value)
        return abbreviations


class UseAcronyms(BaseAbbreviationStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/use_acronyms
    B.110
    """

    def __init__(self, dataset: typing.List[PetDocument]):
        abbreviations = self._load()
        super().__init__(dataset, abbreviations)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @staticmethod
    def _load() -> typing.Dict[str, typing.List[str]]:
        abbreviations = {}

        abbreviations_path = (
            pathlib.Path(__file__)
            .parent.parent.joinpath("resources")
            .joinpath("abbreviations")
            .joinpath("110.tsv")
            .resolve()
        )
        with open(abbreviations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                key, value = line.split("\t", 1)
                if key not in abbreviations:
                    abbreviations[key] = []
                abbreviations[key].append(value)
        return abbreviations


if __name__ == "__main__":

    def main():
        doc = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()[0]
        step = ContractionsAndExpansionsPerturbation([])
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
