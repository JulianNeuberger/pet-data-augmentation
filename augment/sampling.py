import random
import time
import typing

import data
from augment import base, params
from data import PetDocument, PetToken


class EntityMentionReplacement(base.BaseTokenReplacementStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/entity_mention_replacement_ner
    B.39
    """

    def __init__(
        self, dataset: typing.List[PetDocument], replace_probability: float, **kwargs
    ):
        super().__init__(dataset, replace_probability, **kwargs)

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        mentions_token_indices = [m.token_document_indices for m in doc.mentions]
        mention_tokens = [
            [doc.tokens[i] for i in mention_token_indices]
            for mention_token_indices in mentions_token_indices
        ]
        return mention_tokens

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        dataset = [d for d in self.dataset if d != document]
        mention = document.get_mention_for_token(candidate[0])
        print(f'Finding replacements for "{mention.text(document)}" ({mention.type})')
        replacements: typing.List[typing.List[str]] = []
        for _ in range(len(dataset)):
            document_candidate: PetDocument = random.choice(dataset)
            dataset.remove(document_candidate)
            mention_candidates = [
                m for m in document_candidate.mentions if m.type == mention.type
            ]
            for mention_candidate in mention_candidates:
                replacements.append(
                    [
                        document_candidate.tokens[i].text
                        for i in mention_candidate.token_document_indices
                    ]
                )

        random.shuffle(replacements)
        return replacements[:num_replacements_per_candidate]


class TagSubsequenceSubstitution(base.BaseTokenReplacementStep):
    """
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/tag_subsequence_substitution
    B.103
    """

    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replace_probability: float,
        min_n_gram=1,
        max_n_gram=4,
        **kwargs,
    ):
        super().__init__(dataset, replace_probability, **kwargs)
        self.min_n_gram = min_n_gram
        self.max_n_gram = max(self.min_n_gram, max_n_gram)
        self.text_by_pos: typing.Dict[
            typing.Tuple[str, ...], typing.List[typing.Tuple[str, ...]]
        ] = {}
        start_time = time.time_ns()
        for document in self.dataset:
            for sentence in document.sentences:
                for subsequence_length in range(
                    self.min_n_gram, min(len(sentence), self.max_n_gram)
                ):
                    for start in range(0, len(sentence) - subsequence_length):
                        sub_sequence = sentence[start : start + subsequence_length]
                        pos = tuple([t.pos_tag for t in sub_sequence])
                        text = tuple([t.text for t in sub_sequence])
                        if pos not in self.text_by_pos:
                            self.text_by_pos[pos] = []
                        self.text_by_pos[pos].append(text)
        print(
            f"Preprocessing dataset for B.103 took {(time.time_ns() - start_time) / 1e9:.4f}s"
        )

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return base.BaseTokenReplacementStep.get_params() + [
            params.IntegerParam(name="min_n_gram", min_value=1, max_value=10),
            params.IntegerParam(name="max_n_gram", min_value=1, max_value=10),
        ]

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = []
        for segment in self.get_sequences(doc):
            if len(segment) < self.min_n_gram:
                continue

            for subsequence_length in range(self.min_n_gram, self.max_n_gram):
                for start in range(0, len(segment) - subsequence_length):
                    candidate = segment[start : start + subsequence_length]
                    candidate_pos = tuple(t.pos_tag for t in candidate)

                    if candidate_pos in self.text_by_pos:
                        candidates.append(candidate)

        random.shuffle(candidates)
        return candidates

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        candidate_pos = tuple(t.pos_tag for t in candidate)
        possible_replacements = [list(text) for text in self.text_by_pos[candidate_pos]]
        if num_replacements_per_candidate > len(possible_replacements):
            return possible_replacements
        return random.sample(possible_replacements, num_replacements_per_candidate)


if __name__ == "__main__":

    def main():
        docs = data.pet.NewPetFormatImporter(
            r"C:\Users\Neuberger\PycharmProjects\pet-data-augmentation\jsonl\all.new.jsonl"
        ).do_import()
        doc = docs.pop(0)
        step = EntityMentionReplacement(docs)
        augs = step.do_augment(doc, 10)
        print(" ".join(t.text for t in doc.tokens))
        print("-----------")
        for a in augs:
            print(" ".join(t.text for t in a.tokens))
            print()

    main()
