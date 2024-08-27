import typing

import numpy as np

import data
from augment import base, params
from data import PetDocument, PetToken, mutate


class RandomDeletion(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[PetDocument],
        p: float = 1,
    ):
        super().__init__(dataset)
        self.p = p

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
        ]

    def do_augment(
        self, doc: PetDocument, num_augmentations: int
    ) -> typing.List[PetDocument]:
        augmented_documents = []

        for _ in range(num_augmentations):
            aug_doc = doc.copy(clear=[])
            mask = np.random.binomial(1, 1 - self.p, len(aug_doc.tokens)) == 1
            mask = np.flip(mask)
            indices = reversed(list(range(len(aug_doc.tokens))))
            for i, keep in zip(indices, mask):
                if keep:
                    continue
                mutate.delete_token_inplace(aug_doc, i)
            augmented_documents.append(aug_doc)

        return augmented_documents


if __name__ == "__main__":

    def main():
        docs = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()
        doc = docs.pop(0)
        step = RandomDeletion(docs, p=0.25)
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
