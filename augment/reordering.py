import itertools
import typing

import data
from augment import base, params
from data import PetDocument, PetToken


class ShuffleWithinSegments(base.BaseTokenReplacementStep):
    """
    B.90
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/shuffle_within_segments
    """

    def __init__(self, dataset: typing.List[PetDocument], prob: float = 0.5):
        super().__init__(dataset, replacements_per_document=5)
        self.prob = prob

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="prob", min_value=0.0, max_value=1.0),
        ]

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        sequences = self.get_sequences(doc)
        candidates = []
        for sequence in sequences:
            if len(sequence) < 2:
                continue
            # if np.random.binomial(1, self.prob) == 0:
            #   continue
            candidates.append(sequence)
        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        replacements = []
        options = itertools.permutations(candidate)
        for option in options:
            texts = [t.text for t in option]
            original_texts = [t.text for t in candidate]
            if " ".join(texts) != " ".join(original_texts):
                replacements.append(texts)
            if len(replacements) >= num_replacements_per_candidate:
                break
        return replacements


if __name__ == "__main__":

    def main():
        doc = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()[0]
        step = ShuffleWithinSegments([], 0.5)
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
