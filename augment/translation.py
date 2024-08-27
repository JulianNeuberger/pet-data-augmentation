import traceback
import typing

from nltk import tokenize
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

import data.pet
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
        replacements_per_document: int,
        segment_length: int = 2,
        lang: str = "de",
        device: str = "cpu",
    ):
        super().__init__(dataset, replacements_per_document)
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
        return [
            params.IntegerParam(
                name="replacements_per_sentence", min_value=1, max_value=20
            ),
            params.IntegerParam(name="num_beams", min_value=1, max_value=20),
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
        return [tokenize.word_tokenize(t) for t in translated_batch]

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


if __name__ == "__main__":
    doc = data.pet.NewPetFormatImporter(
        r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
    ).do_import()[0]
    step = BackTranslation([], 5, 3, lang="de", device="cpu")
    augs = step.do_augment(doc, 10)
    print(" ".join(t.text for t in doc.tokens))
    print("-----------")
    for a in augs:
        print(" ".join(t.text for t in a.tokens))
        print()
