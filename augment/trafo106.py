import collections
import random
import typing

import nltk
import spacy
import spacy.tokens
import torch
import transformers

from pos_enum import Pos
import data
from augment import base, params
from data import model

POS_TYPES = [
    "ADJ",
    "ADP",
    "PUNCT",
    "ADV",
    "AUX",
    "SYM",
    "INTJ",
    "CONJ",
    "X",
    "NOUN",
    "DET",
    "PROPN",
    "NUM",
    "VERB",
    "PART",
    "PRON",
    "SCONJ",
]

OriginalWord = collections.namedtuple("OriginalWord", ["idx", "text"])


class Trafo106Step(base.AugmentationStep):
    """
    based on
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/transformer_fill/transformation.py
    """

    def __init__(
        self,
        dataset: typing.List[model.Document],
        n: int = 1,
        spacy_model: str = "en_core_web_sm",
        transformer_model: str = "distilroberta-base",
        top_k: int = 5,
        context_text: str = "",
        device: int = -1,
        tag_groups: typing.List[Pos] = None,
        sample_top_k: bool = False,
    ):
        super().__init__(dataset)
        self.n = n
        self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
        self.fill_pipeline = transformers.pipeline(
            "fill-mask", model=transformer_model, top_k=top_k, device=device
        )
        self.pos_tags_to_consider: typing.List[str] = [
            v for group in tag_groups for v in group.tags
        ]
        self.sample_top_k = sample_top_k

        # context text gets prepended to sentence - can be used to prime transformer predictions
        self.context_text = context_text

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_model)
        self.context_length = len(
            self.tokenizer.encode(
                context_text, truncation=True, add_special_tokens=False
            )
        )
        # number of tokens to reserve for expansions of masks
        # (e.g. 1 token gets masked and replaced by 3 tokens)
        self.padding = 32

    def do_augment(self, doc: model.Document) -> model.Document:
        token_texts = [t.text for t in doc.tokens]
        max_token_count = (
            self.tokenizer.model_max_length - self.padding - self.context_length - 4
        )
        token_texts = token_texts[:max_token_count]
        num_tokens_to_truncate = self.tokens_to_truncate(
            " ".join(token_texts), max_token_count
        )
        while num_tokens_to_truncate > 0:
            token_texts = token_texts[:-num_tokens_to_truncate]
            num_tokens_to_truncate = self.tokens_to_truncate(
                " ".join(token_texts), max_token_count
            )

        spacy_doc = spacy.tokens.Doc(vocab=self.nlp.vocab, words=token_texts)
        spacy_doc = self.nlp(spacy_doc, disable=["ner", "lemmatizer"])
        masked_texts, original_words = self.get_masked_sentences_from_sentence(
            spacy_doc
        )
        if len(masked_texts) <= 0:
            return doc.copy()
        predictions = self.fill_pipeline(masked_texts)
        return self.generate_from_predictions(doc, predictions, original_words)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=10),
            params.BooleanParameter(name="sample_top_k"),
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
        ]

    def tokens_to_truncate(self, text: str, max_length: int) -> int:
        """
        Estimates the number of (original) tokens to truncate to fit the model
        """

        encoded = self.tokenizer.encode(
            text,
            truncation=True,
            add_special_tokens=False,
            max_length=self.tokenizer.model_max_length - self.context_length - 4,
        )
        return len(encoded) - max_length

    def get_masked_sentences_from_sentence(
        self, doc: spacy.tokens.Doc
    ) -> typing.Tuple[typing.List[str], typing.List[OriginalWord]]:
        masked_texts = []
        original_words = []
        for token in doc:
            if token.pos_ in self.pos_tags_to_consider:
                masked_texts.append(
                    (
                        self.context_text
                        + doc[: token.i].text
                        + " "
                        + self.fill_pipeline.tokenizer.mask_token
                        + " "
                        + doc[token.i + 1 :].text
                    ).strip()
                )
                original_words.append(OriginalWord(token.i, token.text))

        if len(masked_texts) >= self.n:
            # select n words to replace
            selection = random.sample(list(zip(masked_texts, original_words)), self.n)
            masked_texts, original_words = zip(
                *sorted(selection, key=lambda x: x[1].idx)
            )
            return list(masked_texts), list(original_words)
        return masked_texts, original_words

    def generate_from_predictions(
        self,
        doc: data.Document,
        predictions: typing.List[typing.Any],
        original_words: typing.List[OriginalWord],
    ) -> data.Document:
        doc = doc.copy()
        if len(original_words) < 1:
            return doc
        elif len(original_words) == 1:
            predictions = [predictions]

        # initial text from beginning of sentence to first word to replace
        for i, (preds, original_word) in enumerate(zip(predictions, original_words)):
            predicted_scores = []
            predicted_words = []
            for p in preds:
                if p["token_str"].strip() != original_word.text:
                    predicted_scores.append(p["score"])
                    predicted_words.append(p["token_str"])

            # predict from scores
            if self.sample_top_k:
                x = torch.nn.functional.softmax(torch.tensor(predicted_scores), dim=0)
                cat_dist = torch.distributions.categorical.Categorical(x)
                selected_predicted_index = cat_dist.sample().item()
            else:
                selected_predicted_index = torch.tensor(predicted_scores).argmax()

            selected_prediction = predicted_words[selected_predicted_index]
            token = doc.tokens[original_word.idx]
            token.text = selected_prediction
            token.pos_tag = nltk.pos_tag([selected_prediction])

        return doc
