import random
import typing

import nltk.tokenize
from checklist.editor import Editor

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo86HyponymReplacement(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], n: int = 10):
        super().__init__(dataset)
        self.n = n
        self.editor = Editor()

    def do_augment(
        self,
        doc: model.Document,
    ) -> model.Document:
        doc = doc.copy()
        nouns = [t for t in doc.tokens if t.pos_tag in ["NN", "NNS", "NNP", "NNPS"]]
        random.shuffle(nouns)

        num_changes = 0
        for s in doc.sentences:
            sentence = " ".join(t.text for t in s.tokens)

            nouns = [t for t in s.tokens if t.pos_tag in ["NN", "NNS", "NNP", "NNPS"]]
            random.shuffle(nouns)

            for token in nouns:
                hyponyms = self.editor.hyponyms(sentence, token.text)
                if len(hyponyms) == 0:
                    continue

                hyponym_tokens = nltk.tokenize.word_tokenize(hyponyms[0])
                token_start = token.index_in_sentence(doc)
                tokenmanager.replace_sequence_text_in_sentence(
                    doc, token.sentence_index, token_start, token_start + 1, hyponym_tokens
                )

                num_changes += 1
                if num_changes == self.n:
                    break
        return doc

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [params.IntegerParam(name="n", min_value=1, max_value=20)]


class Trafo86HypernymReplacement(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], n: int = 10):
        super().__init__(dataset)
        self.n = n
        self.editor = Editor()

    def do_augment(
        self,
        doc: model.Document,
    ) -> model.Document:
        doc = doc.copy()

        num_changes = 0
        for s in doc.sentences:
            sentence = " ".join(t.text for t in s.tokens)

            nouns = [t for t in s.tokens if t.pos_tag in ["NN", "NNS", "NNP", "NNPS"]]
            random.shuffle(nouns)

            for token in nouns:
                hyperyms = self.editor.hypernyms(sentence, token.text)
                if len(hyperyms) == 0:
                    continue

                hyponym_tokens = nltk.tokenize.word_tokenize(hyperyms[0])
                token_start = token.index_in_sentence(doc)
                tokenmanager.replace_sequence_text_in_sentence(
                    doc, token.sentence_index, token_start, token_start + 1, hyponym_tokens
                )

                num_changes += 1
                if num_changes == self.n:
                    break
        return doc

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [params.IntegerParam(name="n", min_value=1, max_value=20)]
