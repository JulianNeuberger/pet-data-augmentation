import random
import typing

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo24Step(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], n: int):
        super().__init__(dataset)
        self.n = n

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        for _ in range(self.n):
            num_sentences = len(doc.sentences)
            if num_sentences == 1:
                return doc

            # first index can not be the last sentence!
            first_index = random.randrange(0, num_sentences - 1)

            self.merge_sentences(first_index, doc)
        return doc

    @staticmethod
    def merge_sentences(first_index: int, doc: model.Document):
        second_index = first_index + 1

        first_sentence = doc.sentences[first_index]
        second_sentence = doc.sentences[second_index]

        last_token = first_sentence.tokens[-1]
        tokenmanager.delete_token(
            doc, index_in_document=last_token.index_in_document
        )
        first_sentence_length = len(first_sentence.tokens)

        for sentence in doc.sentences[second_index:]:
            for token in sentence.tokens:
                token.sentence_index -= 1

        for relation in doc.relations:
            evidence = []
            for evidence_sentence_index in relation.evidence:
                new_evidence = evidence_sentence_index
                if evidence_sentence_index >= second_index:
                    new_evidence -= 1
                if new_evidence not in evidence:
                    evidence.append(new_evidence)
            relation.evidence = evidence

        for mention in doc.mentions:
            if mention.sentence_index > second_index:
                mention.sentence_index -= 1
            elif mention.sentence_index == second_index:
                mention.sentence_index -= 1
                mention.token_indices = [
                    t + first_sentence_length for t in mention.token_indices
                ]

        first_sentence.tokens += second_sentence.tokens

        doc.sentences.pop(second_index)
