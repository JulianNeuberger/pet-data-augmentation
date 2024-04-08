import copy
import random
import typing

from transformers import pipeline

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo58Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        languages: typing.List[str],
        strategy: str = "strict",
        num_translations: int = 5,
        n: int = 10,
        device: typing.Optional[int] = 0,
    ):
        super().__init__(dataset)
        self.languages = languages
        self.n = n
        self.strategy = strategy
        self.num_translations = num_translations

        self.encoders = {
            lang: pipeline(
                "translation_en_to_{}".format(lang),
                model="Helsinki-NLP/opus-mt-en-{}".format(lang),
                device=device,
            )
            for lang in self.languages
        }

        self.decoders = {
            lang: pipeline(
                "translation_{}_to_en".format(lang),
                model="Helsinki-NLP/opus-mt-{}-en".format(lang),
                device=device,
            )
            for lang in self.languages
        }

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20),
            params.ChoiceParam(
                name="languages",
                choices=["es", "de", "zh", "fr", "ru"],
                max_num_picks=5,
            ),
            params.ChoiceParam(
                name="strategy", choices=["strict", "shuffle", "random"]
            ),
            params.IntegerParam(name="num_translations", min_value=1, max_value=5),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        return self.do_augment_batch_wise(doc)

    def do_augment_batch_wise(self, doc: model.Document):
        doc = copy.deepcopy(doc)
        translation_candidates = self.get_sequences(doc)
        num_to_translate = min(self.n, len(translation_candidates))
        to_translate = random.sample(translation_candidates, num_to_translate)

        batch = [" ".join([t.text for t in sequence]) for sequence in to_translate]
        translated_batch = self.encode_decode(batch)
        for translated_text, original_sequence in zip(translated_batch, to_translate):
            translated_tokens = translated_text.split()
            sentence_id = original_sequence[0].sentence_index
            start_in_sentence = original_sequence[0].index_in_sentence(doc)
            stop_in_sentence = original_sequence[-1].index_in_sentence(doc) + 1
            tokenmanager.replace_sequence_text_in_sentence(
                doc, sentence_id, start_in_sentence, stop_in_sentence, translated_tokens
            )

        return doc

    def encode_decode(self, texts: typing.List[str]) -> typing.List[str]:
        languages = self.languages
        if self.strategy == "shuffle":
            languages = random.sample(languages, len(languages))
        for i in range(self.num_translations):
            if self.strategy == "random":
                lang = random.choice(languages)
            else:
                lang = languages[i % len(languages)]
            texts = self.back_translate(texts, lang)
        return texts

    def back_translate(self, texts: typing.List[str], language: str):
        encode = self.encoders[language]
        decode = self.decoders[language]
        encoded = [t["translation_text"] for t in encode(texts, max_length=600)]
        decoded = [t["translation_text"] for t in decode(encoded, max_length=600)]
        return decoded
