import random
import typing

import augment
from augment import params
from data import model


class TrafoRandomSwapStep(augment.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], n: int,  **kwargs):
        super().__init__(dataset, **kwargs)
        self.n = n

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        for _ in range(self.n):
            source_index = random.randrange(0, len(doc.tokens))
            target_index = random.randrange(0, len(doc.tokens))
            while target_index == source_index:
                target_index = random.randrange(0, len(doc.tokens))

            source = doc.tokens[source_index]
            target = doc.tokens[target_index]

            source_text = source.text
            source_pos = source.pos_tag

            source.text = target.text
            source.pos_tag = target.pos_tag

            target.text = source_text
            target.pos_tag = source_pos

        return doc

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20)
        ]
