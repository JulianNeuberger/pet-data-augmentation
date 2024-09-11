import random
import traceback
import typing

import nltk
from nltk import tokenize
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import data
from augment import base, params
from data import PetDocument, PetToken


class BackTranslation(base.BaseTokenReplacementStep):
    """
    B.8
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/back_translation
    """

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replace_probability: float,
        segment_length: int = 2,
        lang: str = "de",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(dataset, replace_probability, **kwargs)
        self.device = device
        self.lang = lang
        self.segment_length = segment_length
        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(name_en_de).to(
            device
        )
        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(name_de_en).to(
            device
        )

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return base.BaseTokenReplacementStep.get_params() + [
            params.IntegerParam(name="segment_length", min_value=1, max_value=6),
        ]

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        sequences = self.get_sequences(doc)
        return [s for s in sequences if len(s) > self.segment_length]

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        text = " ".join(t.text for t in candidate)
        if text in [",", ".", "?", "!", ":", "#", "-", "``"]:
            return []

        translated_batch = self.back_translate(text, num_replacements_per_candidate)
        translated_batch = [t for t in translated_batch if t.lower() != text.lower()]
        translated_batch = [t for t in translated_batch if len(t) > 0]
        tokenized = [tokenize.word_tokenize(t) for t in translated_batch]
        zero_length = [t for t in tokenized if len(t) == 0]
        if len(zero_length) > 0:
            print(f"zero length in {tokenized} from {translated_batch}")
        return tokenized

    def back_translate(self, en: str, num_translations: int) -> typing.List[str]:
        try:
            de = self.en2de(en, num_translations)
            return self.de2en(de)
        except Exception as ex:
            print("Ex:")
            print(ex)
            print(traceback.format_exc())
            print("Returning Default due to Run Time Exception")
            return []

    def en2de(self, en: str, num_translations: int) -> typing.List[str]:
        input_ids = self.tokenizer_en_de.encode(en, return_tensors="pt").to(self.device)
        outputs = self.model_en_de.generate(
            input_ids, num_return_sequences=num_translations
        )
        decoded = self.tokenizer_en_de.batch_decode(
            outputs.to(self.device), skip_special_tokens=True
        )
        return decoded

    def de2en(self, de: typing.List[str]) -> typing.List[str]:
        inputs = self.tokenizer_de_en(de, return_tensors="pt", padding=True).to(
            self.device
        )
        outputs = self.model_de_en.generate(
            **inputs,
            num_return_sequences=2,
        )
        return self.tokenizer_de_en.batch_decode(
            outputs.to(self.device), skip_special_tokens=True
        )


class MultiLingualBackTranslation(base.BaseTokenReplacementStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/multilingual_back_translation
    B.62
    """

    languages = [
        "af",
        "am",
        "ar",
        "ast",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "br",
        "bs",
        "ca",
        "ceb",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "ff",
        "fi",
        "fr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "ig",
        "ilo",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "lb",
        "lg",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "my",
        "ne",
        "nl",
        "no",
        "ns",
        "oc",
        "or",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sd",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "ss",
        "su",
        "sv",
        "sw",
        "ta",
        "th",
        "tl",
        "tn",
        "tr",
        "uk",
        "ur",
        "uz",
        "vi",
        "wo",
        "xh",
        "yi",
        "yo",
        "zh",
        "zu",
    ]

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replace_probability: float,
        pivot_language: str = "de",
        **kwargs,
    ):
        super().__init__(dataset, replace_probability, **kwargs)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.src_lang = "en"
        self.pivot_lang = pivot_language

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return base.BaseTokenReplacementStep.get_params() + [
            params.ChoiceParam(
                name="pivot_language", choices=MultiLingualBackTranslation.languages
            ),
        ]

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        return self.get_sequences(doc)

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        sentence = " ".join(t.text for t in candidate)
        candidate_translations = self.back_translate(
            sentence, num_translations=num_replacements_per_candidate
        )
        translation_tokens = []
        for t in candidate_translations:
            translation_tokens.append(tokenize.word_tokenize(t))
        return translation_tokens

    def back_translate(self, sentence: str, num_translations: int):
        pivot_sentences = self.translate(
            [sentence],
            self.src_lang,
            self.pivot_lang,
            self.model,
            self.tokenizer,
            num_translations,
        )
        back_translated = self.translate(
            pivot_sentences,
            self.pivot_lang,
            self.src_lang,
            self.model,
            self.tokenizer,
            2,
        )

        results = []
        for translation in back_translated:
            if translation.lower() != sentence.lower():
                results.append(translation)
        return results

    @staticmethod
    def translate(
        sentences: typing.List[str],
        source_lang: str,
        target_lang: str,
        translation_model,
        tokenizer,
        num_translations: int,
    ) -> typing.List[str]:
        tokenizer.src_lang = source_lang
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        outputs = translation_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            num_return_sequences=num_translations,
        )
        target_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return target_sentences


class LostInTranslation(base.BaseTokenReplacementStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/lost_in_translation/
    B.52
    """

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replace_probability: float,
        languages: typing.List[str],
        strategy: str = "strict",
        num_translation_hops: int = 5,
        device: typing.Optional[int] = 0,
        **kwargs,
    ):
        super().__init__(dataset, replace_probability, **kwargs)
        self.languages = languages
        self.strategy = strategy
        self.num_translation_hops = num_translation_hops

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
        return base.BaseTokenReplacementStep.get_params() + [
            params.ChoiceParam(
                name="languages",
                choices=["es", "de", "zh", "fr", "ru"],
                max_num_picks=5,
            ),
            params.ChoiceParam(
                name="strategy", choices=["strict", "shuffle", "random"]
            ),
            params.IntegerParam(name="num_translation_hops", min_value=1, max_value=5),
        ]

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = self.get_sequences(doc)
        candidates = [c for c in candidates if len(c) > 2]
        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        candidate_text = " ".join(t.text for t in candidate)
        replacement_texts = self.encode_decode(candidate_text)
        return [nltk.tokenize.word_tokenize(t) for t in replacement_texts]

    def encode_decode(self, text: str) -> typing.List[str]:
        languages = self.languages
        translations = [text]
        if self.strategy == "shuffle":
            languages = random.sample(languages, len(languages))
        for i in range(self.num_translation_hops):
            if self.strategy == "random":
                lang = random.choice(languages)
            else:
                lang = languages[i % len(languages)]
            translations = self.back_translate(translations, lang, 5 if i == 0 else 1)
        return translations

    def back_translate(
        self, texts: typing.List[str], language: str, num_translations: int
    ):
        encode = self.encoders[language]
        decode = self.decoders[language]

        encoded = encode(
            texts,
            max_length=600,
            num_return_sequences=num_translations,
            num_beams=num_translations,
        )
        if num_translations > 1:
            encoded = encoded[0]

        encoded = [t["translation_text"] for t in encoded]
        decoded = [t["translation_text"] for t in decode(encoded, max_length=600)]
        return decoded


if __name__ == "__main__":

    def main():
        doc = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()[0]
        step = MultiLingualBackTranslation([], 5, "ms")
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
