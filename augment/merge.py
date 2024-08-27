import random
import typing

from augment import base, params
from data import PetDocument


class MergeDocumentsStep(base.AugmentationStep):
    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        dataset = self.dataset.copy()
        dataset.remove(doc)

        augmented = []

        for _ in range(num_augments):
            if len(dataset) == 0:
                break

            augmented_doc = doc.copy(clear=[])
            other = random.choice(self.dataset)
            dataset = dataset.remove(other)
            augmented_doc.merge(other)
            augmented.append(augmented_doc)

        return augmented
