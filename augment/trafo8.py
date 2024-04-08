import typing

from nltk import tokenize
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from augment import base, params
from data import model


# Author: Benedikt
class Trafo8Step(base.BaseTokenReplacementStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        n: int = 1,
        num_beams: int = 2,
        lang: str = "de",
        device: str = "cpu",
    ):
        super().__init__(dataset, n)
        self.device = device
        self.lang = lang
        self.num_beams = num_beams
        self.max_outputs = 1
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
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20),
            params.IntegerParam(name="num_beams", min_value=1, max_value=20),
        ]

    def get_replacement_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        return self.get_sequences(doc)

    def get_replacement(
        self, candidate: typing.List[model.Token]
    ) -> typing.Optional[typing.List[str]]:
        text = " ".join(t.text for t in candidate)
        if text in [",", ".", "?", "!", ":", "#", "-", "``"]:
            return None

        translated = self.back_translate(text)
        if translated is None:
            return None
        if translated == "":
            return None

        return tokenize.word_tokenize(translated)

    def back_translate(self, en: str) -> typing.Optional[str]:
        try:
            de = self.en2de(en)
            return self.de2en(de)
        except Exception as ex:
            print(ex)
            print("Returning Default due to Run Time Exception")
            return None

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt").to(self.device)
        outputs = self.model_en_de.generate(input_ids)
        decoded = self.tokenizer_en_de.decode(
            outputs[0].to(self.device), skip_special_tokens=True
        )
        return decoded

    def de2en(self, input):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt").to(self.device)
        outputs = self.model_de_en.generate(
            input_ids,
            num_return_sequences=self.max_outputs,
            num_beams=self.num_beams,
        )

        return self.tokenizer_de_en.decode(
            outputs[0].to(self.device), skip_special_tokens=True
        )
