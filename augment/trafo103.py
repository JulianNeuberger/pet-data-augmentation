import random
import time
import typing

from augment import base, params
from data import model


class Trafo103Step(base.AugmentationStep):
    def __init__(
        self, dataset: typing.List[model.Document], n: int, min_n_gram=1, max_n_gram=4
    ):
        super().__init__(dataset)
        self.n = n
        self.min_n_gram = min_n_gram
        self.max_n_gram = max(self.min_n_gram, max_n_gram)
        self.text_by_pos: typing.Dict[
            typing.Tuple[str], typing.List[typing.Tuple[str]]
        ] = {}
        start = time.time_ns()
        for document in self.dataset:
            for sentence in document.sentences:
                for subsequence_length in range(
                    self.min_n_gram, min(len(sentence.tokens), self.max_n_gram)
                ):
                    for start in range(0, len(sentence.tokens) - subsequence_length):
                        sub_sequence = sentence.tokens[
                            start : start + subsequence_length
                        ]
                        pos = tuple([t.pos_tag for t in sub_sequence])
                        text = tuple([t.text for t in sub_sequence])
                        if pos not in self.text_by_pos:
                            self.text_by_pos[pos] = []
                        self.text_by_pos[pos].append(text)
        print(
            f"Preprocessing dataset for B.103 took {(time.time_ns() - start) / 1e6:.4f}ms"
        )

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20),
            params.IntegerParam(name="min_n_gram", min_value=1, max_value=10),
            params.IntegerParam(name="max_n_gram", min_value=1, max_value=10),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        sequences_by_pos_tags: typing.Dict[
            typing.Tuple[str], typing.List[typing.List[str]]
        ]

        doc = doc.copy()

        for i, sentence in enumerate(doc.sentences):
            doc.sentences[i] = self.perturb_sentence(sentence)

        return doc

    def perturb_sentence(self, sentence: model.Sentence) -> model.Sentence:
        for _ in range(self.n):
            if len(sentence.tokens) < self.min_n_gram:
                return sentence

            subsequence_length = random.randint(
                self.min_n_gram, min(len(sentence.tokens), self.max_n_gram)
            )

            start = random.randint(0, len(sentence.tokens) - subsequence_length)
            sub_sequence = sentence.tokens[start : start + subsequence_length]
            pos = tuple([t.pos_tag for t in sub_sequence])

            if pos not in self.text_by_pos:
                continue

            original_text = tuple([t.text for t in sub_sequence])
            new_texts = self.text_by_pos[pos]
            try:
                new_texts.remove(original_text)
            except ValueError:
                pass

            if len(new_texts) == 0:
                continue

            new_text = random.choice(new_texts)

            for token, new_token_text in zip(sub_sequence, new_text):
                token.text = new_token_text

            print(f"Replaced {original_text} with {new_text}")
        return sentence
