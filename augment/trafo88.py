import itertools
import random
import typing

import data
from augment import base, params
from data import PetDocument, mutate


class Trafo88Step(base.AugmentationStep):
    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        swaps = list(itertools.combinations(range(len(doc.sentences)), 2))
        random.shuffle(swaps)

        augmented_docs = []
        for first_sentence, second_sentence in swaps[:num_augments]:
            augmented_doc = doc.copy(clear=[])
            mutate.swap_sentences_inplace(
                augmented_doc, first_sentence, second_sentence
            )
            augmented_docs.append(augmented_doc)

        return augmented_docs

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []


if __name__ == "__main__":

    def main():
        doc = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()[0]
        step = Trafo88Step([])
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
