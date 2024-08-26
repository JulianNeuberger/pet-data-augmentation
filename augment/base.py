import abc
import inspect
import itertools
import random
import typing

import nltk.tokenize

from augment import params
from data import PetDocument, PetToken, PetMention
from transformations import tokenmanager


class AugmentationStep(abc.ABC):
    def __init__(self, dataset: typing.List[PetDocument], **kwargs):
        self.dataset = dataset

    @abc.abstractmethod
    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
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
        doc: PetDocument,
    ) -> typing.List[typing.List[PetToken]]:
        """
        Returns a list of sequences (lists) of tokens, that have
        the same ner tag.
        """
        tagged_sequences = []

        cur_sequence = [doc.tokens[0]]
        last_mention: typing.Optional[PetMention] = None
        for token in doc.tokens[1:]:
            if token.sentence_index != cur_sequence[-1].sentence_index:
                # new sentence started
                tagged_sequences.append(cur_sequence)
                cur_sequence = [token]
            cur_mention = doc.get_mention_for_token(token)
            if cur_mention != last_mention:
                # new mention started
                tagged_sequences.append(cur_sequence)
                cur_sequence = [token]
                last_mention = cur_mention
        if len(cur_sequence) > 0:
            tagged_sequences.append(cur_sequence)

        return tagged_sequences


class BaseTokenReplacementStep(AugmentationStep, abc.ABC):
    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replacements_per_sentence: int,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.replacements_per_sentence = replacements_per_sentence

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @abc.abstractmethod
    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_replacement(
        self, candidate: typing.List[PetToken]
    ) -> typing.Optional[typing.List[str]]:
        raise NotImplementedError()

    def augment_single_doc(
        self, doc: PetDocument, candidates: typing.List[typing.List[PetToken]]
    ) -> PetDocument:
        doc = doc.copy(clear=[])
        for candidate in candidates:
            replacement_texts = self.get_replacement(candidate)
            if replacement_texts is None:
                continue

            num_annotations_before = (
                len(doc.mentions),
                len(doc.entities),
                len(doc.relations),
            )

            print(
                f"Replacing '{' '.join(t.text for t in candidate)}' "
                f"(tokens {candidate[0].index_in_document} - "
                f"{candidate[-1].index_in_document + 1}) "
                f"with '{' '.join(replacement_texts)}'."
            )

            tokenmanager.replace_sequence_inplace(
                doc,
                candidate[0].index_in_document,
                candidate[-1].index_in_document + 1,
                replacement_texts,
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
        return doc

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        all_candidates = self.get_replacement_candidates(doc)
        random.shuffle(all_candidates)

        planned_replacements = [all_candidates]
        if len(all_candidates) > self.replacements_per_sentence:
            planned_replacements = itertools.combinations(
                all_candidates, r=self.replacements_per_sentence
            )

        docs = []
        for planned_replacement in planned_replacements:
            docs.append(self.augment_single_doc(doc, planned_replacement))
            if len(docs) >= num_augments:
                break

        return docs


class AbbreviationStep(AugmentationStep, abc.ABC):
    def __init__(
        self,
        dataset: typing.List[PetDocument],
        abbreviations: typing.Dict[str, str],
        case_sensitive: bool = False,
    ):
        super().__init__(dataset)
        self.case_sensitive = case_sensitive
        self.expansions = abbreviations
        self.contractions = {v: k for k, v in abbreviations.items()}

    def get_contraction_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        return self._get_candidates(self.contractions, doc)

    def get_expansion_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        return self._get_candidates(self.expansions, doc)

    @staticmethod
    def _get_candidates(
        dictionary: typing.Dict[str, str], doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        candidates = []
        candidate: typing.List[PetToken] = []
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

    def do_augment(self, doc: PetDocument, num_augments: int) -> PetDocument:
        doc = doc.copy(clear=[])

        expansion_candidates = self.get_expansion_candidates(doc)
        contraction_candidates = self.get_contraction_candidates(doc)

        self.replace_candidates(doc, expansion_candidates, self.expansions)
        self.replace_candidates(doc, contraction_candidates, self.contractions)

        return doc

    @staticmethod
    def replace_candidates(
        doc: PetDocument,
        candidates: typing.List[typing.List[PetToken]],
        lookup_table: typing.Dict[str, str],
    ):
        for candidate in candidates:
            start = candidate[0].index_in_document
            stop = candidate[-1].index_in_document + 1
            key = " ".join(t.text for t in candidate)
            replace_text = lookup_table[key]
            replace_tokens = nltk.tokenize.word_tokenize(replace_text)
            tokenmanager.replace_sequence_inplace(doc, start, stop, replace_tokens)

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
