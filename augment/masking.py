import random
import typing

import nltk
import spacy
import transformers
from checklist.editor import Editor
from transformers import pipeline

import data
from augment import base, params
from data import PetDocument, PetToken
from pos_enum import Pos


class ContextualMeaningPerturbation(base.BaseTokenReplacementStep):
    """
    B.26
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/contextual_meaning_perturbation
    """

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replace_probability: float,
        tag_groups: typing.List[Pos],
    ):
        super().__init__(dataset, replace_probability=replace_probability)
        self.unmasker = pipeline("fill-mask", model="xlm-roberta-base", top_k=5)
        self.pos_tags_to_consider: typing.List[str] = [
            v.lower() for group in tag_groups for v in group.tags
        ]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return base.BaseTokenReplacementStep.get_params() + [
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
        ]

    @staticmethod
    def mask(sentence: typing.List[str], mask_indices: typing.List[int]) -> str:
        ret = [s for s in sentence]
        for i in mask_indices:
            ret[i] = "<mask>"
        return " ".join(ret)

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates: typing.List[typing.List[PetToken]] = []
        for token in doc.tokens:
            if token.pos_tag.lower() in self.pos_tags_to_consider:
                candidates.append([token])
        random.shuffle(candidates)
        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        original = candidate[0].text
        sentence = document.sentences[candidate[0].sentence_index]
        masked_sentence = self.mask(
            [t.text for t in sentence], [document.token_index_in_sentence(candidate[0])]
        )
        print(masked_sentence)
        unmasking_options = self.unmasker(masked_sentence)
        new_tokens = [o["token_str"] for o in unmasking_options]
        new_tokens = [t for t in new_tokens if t != original]
        new_tokens = [t for t in new_tokens if t.strip() != ""]
        return [[t] for t in new_tokens]


class HyponymReplacement(base.BaseTokenReplacementStep):
    """
    B.86
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/replace_with_hyponyms_hypernyms
    """

    def __init__(self, dataset: typing.List[PetDocument], replace_probability: float):
        super().__init__(dataset, replace_probability=replace_probability)
        self.editor = Editor()

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        nouns = []

        for token in doc.tokens:
            if token.pos_tag not in ["NN", "NNS", "NNP", "NNPS"]:
                continue
            sentence = " ".join(t.text for t in doc.sentences[token.sentence_index])
            try:
                hyponyms = self.editor.hyponyms(sentence, token.text)
            except IndexError:
                # happens, if the original word is deemed unlikely by the LM
                continue
            if len(hyponyms) == 0:
                continue
            nouns.append(token)

        random.shuffle(nouns)
        return [[n] for n in nouns]

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        assert len(candidate) == 1
        token = candidate[0]
        sentence = " ".join(t.text for t in document.sentences[token.sentence_index])

        hyponyms = self.editor.hyponyms(sentence, token.text)
        hyponyms = hyponyms[:num_replacements_per_candidate]
        return [nltk.tokenize.word_tokenize(h) for h in hyponyms]


class HypernymReplacement(base.BaseTokenReplacementStep):
    """
    B.86
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/replace_with_hyponyms_hypernyms
    """

    def __init__(self, dataset: typing.List[PetDocument], replace_probability: float):
        super().__init__(dataset, replace_probability=replace_probability)
        self.editor = Editor()

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        nouns = []

        for token in doc.tokens:
            if token.pos_tag not in ["NN", "NNS", "NNP", "NNPS"]:
                continue
            sentence = " ".join(t.text for t in doc.sentences[token.sentence_index])
            try:
                hypernyms = self.editor.hypernyms(sentence, token.text)
            except IndexError:
                # happens, if the original word is deemed unlikely by the LM
                continue
            if len(hypernyms) == 0:
                continue
            nouns.append(token)

        random.shuffle(nouns)
        return [[n] for n in nouns]

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        assert len(candidate) == 1
        token = candidate[0]
        sentence = " ".join(t.text for t in document.sentences[token.sentence_index])
        hypernyms = self.editor.hypernyms(sentence, token.text)
        hypernyms = hypernyms[:num_replacements_per_candidate]
        return [nltk.tokenize.word_tokenize(h) for h in hypernyms]


class TransformerFill(base.BaseTokenReplacementStep):
    """
    B.106
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/transformer_fill/
    """

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replace_probability: float = 0.5,
        spacy_model: str = "en_core_web_sm",
        transformer_model: str = "distilroberta-base",
        device: int = -1,
        tag_groups: typing.List[Pos] = None,
    ):
        super().__init__(dataset, replace_probability=replace_probability)
        self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
        self.fill_pipeline = transformers.pipeline(
            "fill-mask", model=transformer_model, device=device
        )
        self.pos_tags_to_consider: typing.List[str] = [
            v for group in tag_groups for v in group.tags
        ]

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_model)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return base.BaseTokenReplacementStep.get_params() + [
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
        ]

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = []
        for token in doc.tokens:
            if token.pos_tag in self.pos_tags_to_consider:
                candidates.append([token])
        random.shuffle(candidates)
        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        assert len(candidate) == 1

        sentence = document.sentences[candidate[0].sentence_index]
        masked_sentence = [t.text for t in sentence]
        masked_sentence[document.token_index_in_sentence(candidate[0])] = (
            self.fill_pipeline.tokenizer.mask_token
        )

        predictions = self.fill_pipeline(
            " ".join(masked_sentence),
            top_k=num_replacements_per_candidate,
        )

        predicted_scores = []
        predicted_words = []
        for p in predictions:
            if p["token_str"].strip() != candidate[0].text:
                predicted_scores.append(p["score"])
                predicted_words.append(p["token_str"])

        return [nltk.tokenize.word_tokenize(w) for w in predicted_words]


if __name__ == "__main__":

    def main():
        doc = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()[0]
        step = TransformerFill([], 0.25, tag_groups=[Pos.NOUN])
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
