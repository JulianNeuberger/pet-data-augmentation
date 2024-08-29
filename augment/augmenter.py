import random
import typing

import numpy as np
import tqdm

from augment import base
from data import PetDocument


class Augmenter:
    def __init__(
        self, steps: typing.List[base.AugmentationStep], num_augmentations: int
    ):
        self._steps = steps
        self._num_augmentations = num_augmentations
        self._augmented_docs: typing.Dict[str, typing.List[PetDocument]] = {}

    def augment(self, document: PetDocument) -> typing.List[PetDocument]:
        if document.name not in self._augmented_docs:
            self._augmented_docs[document.name] = self._run_augmentation(document)
        return [d.copy(clear=[]) for d in self._augmented_docs[document.name]]

    def _run_augmentation(
        self,
        document: PetDocument,
    ) -> typing.List[PetDocument]:
        self._augmented_docs[document.name] = []
        # apply first step with the actual augmentation rate
        augmented_documents = self._steps[0].do_augment(
            document, self._num_augmentations
        )
        augmented_documents = augmented_documents[: self._num_augmentations]

        # apply all remaining steps on each augmented doc,
        # but only create one augmented document
        for augmented_doc in augmented_documents:
            for step in self._steps[1:]:
                augmented_doc = step.do_augment(augmented_doc, 1)
            self._augmented_docs[document.name].append(augmented_doc)
        return self._augmented_docs[document.name]


def get_augmentation_rates(num_documents: int, augmentation_rate: float) -> np.ndarray:
    augmentation_rates_per_document = np.zeros(num_documents, dtype=int)

    if augmentation_rate >= 1.0:
        augmentation_rates_per_document += int(augmentation_rate)
        augmentation_rate -= int(augmentation_rate)

    num_additional_augmentations = int(augmentation_rate * num_documents)
    additional_augmentation_indices = np.random.choice(
        np.arange(num_documents), size=num_additional_augmentations, replace=False
    )
    augmentation_rates_per_document[additional_augmentation_indices] += 1
    return augmentation_rates_per_document


def run_augmentation(
    dataset: typing.List[PetDocument],
    steps: typing.List[base.AugmentationStep],
    augmentation_rate: float,
) -> typing.List[PetDocument]:
    augmentation_rate_per_document = get_augmentation_rates(
        len(dataset), augmentation_rate
    )

    print(
        f"Augmenting {len(dataset)} documents with "
        f"augmentation factor of {augmentation_rate:.4f} "
        f"resulting in {np.sum(augmentation_rate_per_document)} new documents "
        f"using strategies {[type(s).__name__ for s in steps]}..."
    )

    augmented_dataset: typing.List[PetDocument] = []

    for i, document in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        current_aug_rate = augmentation_rate_per_document[i]
        # apply first step with the actual augmentation rate
        augmented_docs = steps[0].do_augment(document, current_aug_rate)

        # apply all remaining steps on each augmented doc,
        # but only create one augmented document
        for augmented_doc in augmented_docs:
            for step in steps[1:]:
                augmented_doc = step.do_augment(augmented_doc, 1)
            augmented_dataset.append(augmented_doc)

    # add original dataset
    augmented_dataset.extend([d.copy(clear=[]) for d in dataset])

    # finally, shuffle the augmented dataset
    random.shuffle(augmented_dataset)

    return augmented_dataset
