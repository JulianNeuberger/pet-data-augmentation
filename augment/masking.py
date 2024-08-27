import random
import typing

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
        n=1,
        tag_groups: typing.List[Pos] = None,
    ):
        super().__init__(dataset, replacements_per_document=5)
        self.n = n
        self.unmasker = pipeline("fill-mask", model="xlm-roberta-base", top_k=5)
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


if __name__ == "__main__":

    def main():
        doc = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()[0]
        step = ContextualMeaningPerturbation([], 5, tag_groups=[Pos.NOUN])
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
