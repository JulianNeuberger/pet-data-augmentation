import itertools
import random
import typing

from nltk.corpus import wordnet

from augment import base, params
from data import PetDocument, PetToken
from data import mutate


class AntonymInversionStep(base.BaseTokenReplacementStep):
    """
    B.3
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/adjectives_antonyms_switch
    """

    @staticmethod
    def get_default_configuration(
        dataset: typing.List[PetDocument],
    ) -> "AntonymInversionStep":
        return AntonymInversionStep(dataset, replace_probability=0.93)

    def get_replacement_candidates(
        self, document: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = [[t] for t in document.tokens if t.pos_tag in ["JJ", "JJR", "JJS"]]
        candidates = [c for c in candidates if len(self.antonym(c)) > 0]
        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        return self.antonym(candidate)

    @staticmethod
    def antonym(candidate: typing.List[PetToken]):
        text = " ".join(t.text for t in candidate)

        syn_sets = wordnet.synsets(text, "a")
        syn_sets = [s for s in syn_sets if ".a." in s.name()]

        if len(syn_sets) == 0:
            return []

        first_syn_set = syn_sets[0]
        lemma = first_syn_set.lemmas()[0]
        antonyms = lemma.antonyms()

        if len(antonyms) == 0:
            return []

        antonyms.sort(key=lambda x: str(x).split(".")[2])
        antonym: str = antonyms[0].name()
        return [antonym.split("_")]


class EvenAntonymsSubstitute(base.AugmentationStep):
    """
    B.5
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/antonyms_substitute
    """

    @staticmethod
    def get_default_configuration(
        dataset: typing.List[PetDocument],
    ) -> "EvenAntonymsSubstitute":
        return EvenAntonymsSubstitute(dataset)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        # Choose from Adjectives and Adverbs
        pos_tags = ["JJ", "JJS", "JJR", "RB", "RBR", "RBS"]

        candidates: typing.List[typing.Tuple[PetToken, PetToken]] = []
        for sentence in doc.sentences:
            sentence_candidates = [t for t in sentence if t.pos_tag in pos_tags]

            for left, right in itertools.combinations(sentence_candidates, 2):
                if self.are_antonym(left.text, right.text):
                    continue
                if self.are_synonym(left.text, right.text):
                    continue
                left_antonym = self.get_antonym(left)
                if len(left_antonym) == 0:
                    continue

                right_antonym = self.get_antonym(right)
                if len(right_antonym) == 0:
                    continue
                candidates.append((left, right))

        if len(candidates) == 0:
            print(f'No candidates in document with text "{doc.text}"')

        augmented_docs = []
        for _ in range(num_augments):
            if len(candidates) == 0:
                return augmented_docs
            augmented_doc = doc.copy(clear=[])
            left: PetToken
            right: PetToken
            candidate = random.choice(candidates)
            candidates.remove(candidate)
            left, right = candidate
            left_antonym = self.get_antonym(left)
            mutate.replace_sequence_inplace(
                augmented_doc,
                left.index_in_document,
                left.index_in_document + 1,
                left_antonym,
            )

            right_antonym = self.get_antonym(right)
            mutate.replace_sequence_inplace(
                augmented_doc,
                right.index_in_document,
                right.index_in_document + 1,
                right_antonym,
            )
            augmented_docs.append(augmented_doc)

        return augmented_docs

    @staticmethod
    def get_antonym(token: PetToken) -> typing.List[str]:
        syn_sets = wordnet.synsets(token.text, "a")
        syn_sets = [s for s in syn_sets if ".a." in s.name()]

        if len(syn_sets) == 0:
            return []

        first_syn_set = syn_sets[0]
        lemma = first_syn_set.lemmas()[0]
        antonyms = lemma.antonyms()

        if len(antonyms) == 0:
            return []

        antonyms.sort(key=lambda x: str(x).split(".")[2])
        antonym = antonyms[0].name()

        return antonym.split("_")

    @staticmethod
    def contains_antonyms_or_synonyms(tokens):
        for left, right in itertools.combinations(tokens, 2):
            if EvenAntonymsSubstitute.are_antonym(left, right):
                return True
            if EvenAntonymsSubstitute.are_synonym(left, right):
                return True
        return False

    @staticmethod
    def are_antonym(left: str, right: str):
        antonyms = []
        for syn in wordnet.synsets(left):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        return right in antonyms

    @staticmethod
    def are_synonym(left: str, right: str):
        synonyms = []
        for syn in wordnet.synsets(left):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return right in synonyms
