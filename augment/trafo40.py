import random
import typing

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo40Step(base.AugmentationStep):
    """
    Based on https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/filler_word_augmentation/transformation.py
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
        dataset: typing.List[model.Document],
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

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        phrases = self.get_phrases()

        if len(phrases) == 0:
            return doc

        for _ in range(self.n):
            index = random.randrange(0, len(doc.tokens))
            phrase = random.choice(phrases)
            phrase_tokens = phrase.split()

            for phrase_token_id, phrase_token in enumerate(phrase_tokens):
                tokenmanager.insert_token_text_into_document(
                    doc, phrase_token, index + phrase_token_id
                )
        return doc
