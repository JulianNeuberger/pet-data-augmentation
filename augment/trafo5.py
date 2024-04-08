import itertools
import typing

from nltk.corpus import wordnet

from augment import base, params
from data import model

# Author: Benedikt
from transformations import tokenmanager


class Trafo5Step(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], n: int = 1):
        super().__init__(dataset)
        self.n = n

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [params.IntegerParam(name="n", min_value=1, max_value=20)]

    def switch_antonyms_in_sentence(self, sentence: model.Sentence):
        pass

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        # Choose from Adjectives and Adverbs
        pos_tags = ["JJ", "JJS", "JJR", "RB", "RBR", "RBS"]
        # for all sentences and all tokens in the Document

        candidates: typing.List[typing.Tuple[model.Token, model.Token]] = []
        for sentence in doc.sentences:
            sentence_candidates = [t for t in sentence.tokens if t.pos_tag in pos_tags]
            candidates.extend(itertools.combinations(sentence_candidates, 2))

        num_changes = 0
        for left, right in candidates:
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

            tokenmanager.expand_token(doc, left, left_antonym)
            tokenmanager.expand_token(doc, right, right_antonym)

            num_changes += 1
            if num_changes == self.n:
                break
        return doc

    @staticmethod
    def get_antonym(token: model.Token) -> typing.List[str]:
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
            if Trafo5Step.are_antonym(left, right):
                return True
            if Trafo5Step.are_synonym(left, right):
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
