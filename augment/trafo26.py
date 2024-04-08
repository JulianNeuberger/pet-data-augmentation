import random
import typing

from transformers import pipeline

from augment import base, params
from data import model
from pos_enum import Pos
from transformations import tokenmanager


class Trafo26Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        n=1,
        tag_groups: typing.List[Pos] = None,
    ):
        super().__init__(dataset)
        self.n = n
        self.unmasker = pipeline("fill-mask", model="xlm-roberta-base", top_k=1)
        self.pos_tags_to_consider: typing.List[str] = [
            v.lower() for group in tag_groups for v in group.tags
        ]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
            params.IntegerParam(name="n", min_value=1, max_value=20),
        ]

    @staticmethod
    def mask_sentence(
        doc: model.Document, sentence: model.Sentence, token_to_mask: model.Token
    ) -> str:
        tokens = [t.text for t in sentence.tokens]
        tokens[token_to_mask.index_in_sentence(doc)] = "<mask>"
        return " ".join(tokens)

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        candidates: typing.List[model.Token] = []
        for token in doc.tokens:
            if token.pos_tag.lower() in self.pos_tags_to_consider:
                candidates.append(token)
        random.shuffle(candidates)

        num_changes = 0
        for candidate in candidates:
            original = candidate.text
            sentence = doc.sentences[candidate.sentence_index]
            masked_sentence = self.mask_sentence(doc, sentence, candidate)
            new_token = self.unmasker(masked_sentence)[0]["token_str"]

            if new_token == original:
                continue

            if new_token == '':
                continue

            doc.tokens[candidate.index_in_document].text = new_token
            doc.tokens[candidate.index_in_document].pos_tag = tokenmanager.get_pos_tag([new_token])[0]

            num_changes += 1
            if num_changes == self.n:
                break

        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.text.strip() == '':
                    print(f"Erroneous document: {' '.join(t.text for t in doc.tokens)}")

        return doc
