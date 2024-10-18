import random
import typing

import spacy

from augment import base, params
from data import mutate, PetDocument, PetToken


class AuxiliaryNegationRemoval(base.AugmentationStep):
    """
    B.6
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/auxiliary_negation_removal
    """

    def __init__(self, dataset: typing.List[PetDocument]):
        super().__init__(dataset)
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def get_default_configuration(
        dataset: typing.List[PetDocument],
    ) -> "AuxiliaryNegationRemoval":
        return AuxiliaryNegationRemoval(dataset)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        doc = doc.copy(clear=[])
        candidates: typing.List[PetToken] = []
        for token in doc.tokens:
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
                mutate.delete_token_inplace(doc, token.index_in_document)
                continue

            spacy_document = self.nlp(token.text)

            if spacy_document[1].text == "n't":
                doc.tokens[token.index_in_document] = PetToken(
                    pos_tag=token.pos_tag,
                    sentence_index=token.sentence_index,
                    text=spacy_document[0].text,
                    index_in_document=token.index_in_document,
                )

        return [doc]
