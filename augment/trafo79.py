import typing

import numpy as np

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo79Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        p: float = 1,
    ):
        super().__init__(dataset)
        self.p = p

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        mask = np.random.binomial(1, 1 - self.p, len(doc.tokens)) == 1
        mask = np.flip(mask)
        indices = reversed(list(range(len(doc.tokens))))
        for i, keep in zip(indices, mask):
            if keep:
                continue

            tokenmanager.delete_token(doc, i)

        return doc
