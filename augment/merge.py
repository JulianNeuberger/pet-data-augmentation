import random
import typing

from augment import base, params
from data import PetDocument


class MergeDocumentsStep(base.AugmentationStep):
    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @staticmethod
    def get_default_configuration(
        dataset: typing.List[PetDocument],
    ) -> "MergeDocumentsStep":
        return MergeDocumentsStep(dataset)

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        dataset = self.dataset.copy()
        if doc in dataset:
            dataset.remove(doc)

        augmented = []

        for _ in range(num_augments):
            if len(dataset) == 0:
                break

            augmented_doc = doc.copy(clear=[])
            other = random.choice(dataset)
            dataset.remove(other)
            augmented_doc = augmented_doc.merge(other)
            augmented.append(augmented_doc)

        return augmented
