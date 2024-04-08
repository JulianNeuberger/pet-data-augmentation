import typing

from nltk import tokenize
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from augment import base, params
from data import model


class Trafo62Step(base.BaseTokenReplacementStep):
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
        dataset: typing.List[model.Document],
        n: int = 10,
        pivot_language: str = "de",
    ):
        super().__init__(dataset, n)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.src_lang = "en"
        self.pivot_lang = pivot_language

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.ChoiceParam(name="pivot_language", choices=Trafo62Step.languages),
            params.IntegerParam(name="n", min_value=1, max_value=20),
        ]

    def get_replacement_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        return self.get_sequences(doc)

    def get_replacement(
        self, candidate: typing.List[model.Token]
    ) -> typing.Optional[typing.List[str]]:
        sentence = " ".join(t.text for t in candidate)
        translated_candidate = self.back_translate(sentence)[0]
        translated_tokens = tokenize.word_tokenize(translated_candidate)
        return translated_tokens

    def back_translate(self, sentence: str):
        pivot_sentence = self.translate(
            sentence, self.src_lang, self.pivot_lang, self.model, self.tokenizer
        )
        return self.translate(
            pivot_sentence, self.pivot_lang, self.src_lang, self.model, self.tokenizer
        )

    @staticmethod
    def translate(
        sentence: str, source_lang: str, target_lang: str, translation_model, tokenizer
    ) -> str:
        tokenizer.src_lang = source_lang
        encoded_source_sentence = tokenizer(sentence, return_tensors="pt")
        generated_target_tokens = translation_model.generate(
            **encoded_source_sentence,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang)
        )
        target_sentence = tokenizer.batch_decode(
            generated_target_tokens, skip_special_tokens=True
        )
        return target_sentence
