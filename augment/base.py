import abc
import inspect
import random
import typing

import nltk.tokenize

from augment import params
from data import model
from transformations import tokenmanager


class AugmentationStep(abc.ABC):
    def __init__(self, dataset: typing.List[model.Document], **kwargs):
        self.dataset = dataset

    @abc.abstractmethod
    def do_augment(self, doc: model.Document) -> model.Document:
        raise NotImplementedError()

    @staticmethod
    def validate_params(clazz: typing.Type["AugmentationStep"]):
        args = inspect.getfullargspec(clazz.__init__).args
        missing_args = []
        for param in clazz.get_params():
            found_arg = False
            for arg in args:
                if param.name == arg:
                    found_arg = True
                    break
            if not found_arg:
                missing_args.append(param)
        if len(missing_args) > 0:
            raise TypeError(
                f"Missing arguments in __init__ method of {clazz.__name__}: {missing_args}"
            )

    @staticmethod
    @abc.abstractmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        raise NotImplementedError()

    @staticmethod
    def get_sequences(
        doc: model.Document,
    ) -> typing.List[typing.List[model.Token]]:
        """
        Returns a list of sequences (lists) of tokens, that have
        the same ner tag.
        """
        tagged_sequences = []

        for sentence in doc.sentences:
            last_sequence: typing.List[model.Token] = []
            last_mention: typing.Optional[model.Mention] = None
            for token in sentence.tokens:
                mentions = doc.get_mentions_for_token(token)
                if len(mentions) == 0:
                    current_mention = None
                else:
                    current_mention = mentions[0]
                if current_mention != last_mention:
                    if len(last_sequence) > 0:
                        tagged_sequences.append(last_sequence)
                    last_sequence = []
                last_mention = current_mention
                last_sequence.append(token)
            if len(last_sequence) > 0:
                tagged_sequences.append(last_sequence)

        return tagged_sequences


class BaseTokenReplacementStep(AugmentationStep, abc.ABC):
    def __init__(self, dataset: typing.List[model.Document], n: int):
        super().__init__(dataset)
        self.n = n

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [params.IntegerParam(name="n", min_value=1, max_value=20)]

    @abc.abstractmethod
    def get_replacement_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_replacement(
        self, candidate: typing.List[model.Token]
    ) -> typing.Optional[typing.List[str]]:
        raise NotImplementedError()

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        candidates = self.get_replacement_candidates(doc)
        random.shuffle(candidates)

        num_changed = 0
        for candidate in candidates:
            replacement_tokens = self.get_replacement(candidate)
            if replacement_tokens is None:
                continue

            num_annotations_before = (
                len(doc.mentions),
                len(doc.entities),
                len(doc.relations),
            )

            print(f"Replacing '{' '.join(t.text for t in candidate)}' "
                  f"(sentence {candidate[0].sentence_index}, "
                  f"tokens {candidate[0].index_in_sentence(doc)} - "
                  f"{candidate[-1].index_in_sentence(doc) + 1}) "
                  f"with '{' '.join(replacement_tokens)}'.")

            tokenmanager.replace_sequence_text_in_sentence(
                doc,
                candidate[0].sentence_index,
                candidate[0].index_in_sentence(doc),
                candidate[-1].index_in_sentence(doc) + 1,
                replacement_tokens,
            )

            num_annotations_after = (
                len(doc.mentions),
                len(doc.entities),
                len(doc.relations),
            )

            assert num_annotations_before == num_annotations_after, (
                f"Replacing candidates changed the documents annotations! "
                f"This must not happen! Before augmentation there were {num_annotations_before} "
                f"mentions, entities and relations, now there are {num_annotations_after}."
            )

            num_changed += 1
            if num_changed == self.n:
                break

        return doc


class AbbreviationStep(AugmentationStep, abc.ABC):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        abbreviations: typing.Dict[str, str],
        case_sensitive: bool = False,
    ):
        super().__init__(dataset)
        self.case_sensitive = case_sensitive
        self.expansions = abbreviations
        self.contractions = {v: k for k, v in abbreviations.items()}

    def get_contraction_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        return self._get_candidates(self.contractions, doc)

    def get_expansion_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        return self._get_candidates(self.expansions, doc)

    @staticmethod
    def _get_candidates(
        dictionary: typing.Dict[str, str], doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        candidates = []
        candidate: typing.List[model.Token] = []
        for token in doc.tokens:
            candidate += [token]
            candidate_key = " ".join(t.text for t in candidate)
            if candidate_key in dictionary:
                candidates.append(candidate)
                candidate = []
                continue
            if AbbreviationStep.has_keys_starting_with(dictionary, candidate_key):
                candidate += [token]
                continue
            candidate = []
        return candidates

    @staticmethod
    def has_keys_starting_with(
        dictionary: typing.Dict[str, typing.Any], partial_key: str
    ) -> bool:
        for key in dictionary.keys():
            if key.startswith(partial_key):
                return True
        return False

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        expansion_candidates = self.get_expansion_candidates(doc)
        contraction_candidates = self.get_contraction_candidates(doc)

        self.replace_candidates(doc, expansion_candidates, self.expansions)
        self.replace_candidates(doc, contraction_candidates, self.contractions)

        return doc

    @staticmethod
    def replace_candidates(
        doc: model.Document,
        candidates: typing.List[typing.List[model.Token]],
        lookup_table: typing.Dict[str, str],
    ):
        for candidate in candidates:
            start = candidate[0].index_in_sentence(doc)
            stop = candidate[-1].index_in_sentence(doc) + 1
            key = " ".join(t.text for t in candidate)
            replace_text = lookup_table[key]
            replace_tokens = nltk.tokenize.word_tokenize(replace_text)
            tokenmanager.replace_sequence_text_in_sentence(
                doc, candidate[0].sentence_index, start, stop, replace_tokens
            )

    def load_bank110(self):
        sep = "\t"
        temp_acronyms = []
        contracted = []
        expanded = []
        with open(
            "./transformations/trafo82/acronyms.tsv", "r", encoding="utf-8"
        ) as file:
            for line in file:
                key, value = line.strip().split(sep)
                # temp_acronyms[key] = value
                contracted.append(key)
                expanded.append(value)
        # Place long keys first to prevent overlapping
        acronyms = {}
        for k in sorted(temp_acronyms, key=len, reverse=True):
            acronyms[k] = temp_acronyms[k]
        acronyms = acronyms
        return contracted, expanded
