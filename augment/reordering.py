import itertools
import random
import typing

import data
from augment import base, params
from data import PetDocument, PetToken, mutate


class ShuffleWithinSegments(base.BaseTokenReplacementStep):
    """
    B.90
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/shuffle_within_segments
    """

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


class SentenceReordering(base.AugmentationStep):
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


class RandomTokenSwap(base.AugmentationStep):
    def __init__(
        self, dataset: typing.List[PetDocument], swap_probability: float, **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.p = swap_probability

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        augmented_docs = []

        for _ in range(num_augments):
            augmented_doc = doc.copy(clear=[])
            for _ in range(len(augmented_doc.tokens)):
                if random.random() > self.p:
                    continue
                source_index = random.randrange(0, len(augmented_doc.tokens))
                target_index = random.randrange(0, len(augmented_doc.tokens))
                while target_index == source_index:
                    target_index = random.randrange(0, len(augmented_doc.tokens))

                source = augmented_doc.tokens[source_index]
                target = augmented_doc.tokens[target_index]

                augmented_doc.tokens[source_index] = PetToken(
                    pos_tag=target.pos_tag,
                    text=target.text,
                    index_in_document=source.index_in_document,
                    sentence_index=source.sentence_index,
                )
                augmented_doc.tokens[target_index] = PetToken(
                    pos_tag=source.pos_tag,
                    text=source.text,
                    index_in_document=target.index_in_document,
                    sentence_index=target.sentence_index,
                )
            augmented_docs.append(augmented_doc)
        return augmented_docs

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="swap_probability", min_value=0.0, max_value=1.0)
        ]


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
