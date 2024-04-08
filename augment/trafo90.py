import random
import typing

import numpy as np
from numpy.random import binomial

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo90Step(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], prob: float = 0.5):
        super().__init__(dataset)
        self.prob = prob

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="prob", min_value=0.0, max_value=1.0),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        sequences = self.get_sequences(doc)
        for sequence in sequences:
            if np.random.binomial(1, self.prob) == 1:
                # shuffle this sequence
                sequence_sentence_index = sequence[0].sentence_index
                sequence_indices = [t.index_in_sentence(doc) for t in sequence]
                random.shuffle(sequence_indices)

                is_first_in_sequence = True
                for new_index, token in zip(sequence_indices, sequence):
                    if token.bio_tag != "O":
                        if is_first_in_sequence:
                            token.bio_tag = (
                                f"B-{tokenmanager.get_bio_tag_short(token.bio_tag)}"
                            )
                            is_first_in_sequence = False
                        else:
                            token.bio_tag = (
                                f"I-{tokenmanager.get_bio_tag_short(token.bio_tag)}"
                            )
                    doc.sentences[sequence_sentence_index].tokens[new_index] = token
        return doc
