import random
import typing

import data
from augment import base, params
from data import PetDocument, PetToken, mutate


class FillerWordAugmentation(base.AugmentationStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/filler_word_augmentation
    B.40
    """

    # Speaker opinion/mental state phrases
    # Taken from Kovatchev et al. (2021)
    speaker_phrases = [
        "I think",
        "I believe",
        "I mean",
        "I guess",
        "that is",
        "I assume",
        "I feel",
        "In my opinion",
        "I would say",
    ]

    # Words and phrases indicating uncertainty
    # Taken from Kovatchev et al. (2021)
    uncertain_phrases = [
        "maybe",
        "perhaps",
        "probably",
        "possibly",
        "most likely",
    ]

    # Filler words that should preserve the meaning of the phrase
    # Taken from Laserna et al. (2014)
    fill_phrases = [
        "uhm",
        "umm",
        "ahh",
        "err",
        "actually",
        "obviously",
        "naturally",
        "like",
        "you know",
    ]

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        n: int = 10,
        insert_speaker_phrases: bool = True,
        insert_uncertainty_phrases: bool = True,
        insert_filler_phrases: bool = True,
    ):
        super().__init__(dataset)
        self.n = n
        self.insert_speaker_phrases = insert_speaker_phrases
        self.insert_uncertainty_phrases = insert_uncertainty_phrases
        self.insert_filler_phrases = insert_filler_phrases

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20),
            params.BooleanParameter(name="insert_speaker_phrases"),
            params.BooleanParameter(name="insert_uncertainty_phrases"),
            params.BooleanParameter(name="insert_filler_phrases"),
        ]

    def get_phrases(self):
        all_fill = []
        if self.insert_speaker_phrases:
            all_fill += self.speaker_phrases
        if self.insert_uncertainty_phrases:
            all_fill += self.uncertain_phrases
        if self.insert_filler_phrases:
            all_fill += self.fill_phrases
        return all_fill

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        phrases = self.get_phrases()
        assert len(phrases) > 0

        augmented_documents = []
        for _ in range(num_augments):
            augmented_doc = doc.copy(clear=[])
            for _ in range(self.n):
                index = random.randrange(0, len(augmented_doc.tokens))
                phrase = random.choice(phrases)
                phrase_texts = phrase.split()
                mutate.insert_texts_inplace(augmented_doc, phrase_texts, index)
            augmented_documents.append(augmented_doc)

        return augmented_documents


if __name__ == "__main__":

    def main():
        docs = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()
        doc = docs.pop(0)
        step = FillerWordAugmentation(docs)
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
