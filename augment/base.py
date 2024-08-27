import abc
import inspect
import itertools
import random
import typing

import nltk.tokenize

from augment import params
from data import PetDocument, PetToken, PetMention, mutate


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
            cur_mention = doc.get_mention_for_token(token)

            if token.sentence_index != cur_sequence[-1].sentence_index:
                # new sentence started
                tagged_sequences.append(cur_sequence)
                cur_sequence = [token]
            elif cur_mention != last_mention:
                # new mention started
                tagged_sequences.append(cur_sequence)
                cur_sequence = [token]
            else:
                cur_sequence.append(token)

            last_mention = cur_mention
        if len(cur_sequence) > 0:
            tagged_sequences.append(cur_sequence)

        return tagged_sequences


class BaseTokenReplacementStep(AugmentationStep, abc.ABC):
    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replacements_per_document: int,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.replacements_per_document = replacements_per_document

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @abc.abstractmethod
    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        raise NotImplementedError()

    def augment_single_doc(
        self,
        doc: PetDocument,
        candidates: typing.List[typing.List[PetToken]],
        num_augments: int,
        max_replacements_per_candidate: int,
    ) -> typing.List[PetDocument]:
        num_annotations_before = (
            len(doc.mentions),
            len(doc.entities),
            len(doc.relations),
        )

        if len(candidates) == 0:
            print(
                f'Replacement strategy "{self.__class__.__name__}" did not find any candidates!'
            )
        candidate_replacements: typing.Dict[
            typing.Tuple[PetToken, ...], typing.List[typing.List[str]]
        ] = {}
        for candidate in candidates:
            replacement_texts = self.get_replacements(
                candidate, max_replacements_per_candidate, doc
            )
            # remove identical replacements
            replacement_texts = [
                t
                for t in replacement_texts
                if " ".join(c.text for c in candidate) != " ".join(r for r in t)
            ]
            if len(replacement_texts) == 0:
                print(
                    f'Replacement strategy "{self.__class__.__name__}" did not generate any '
                    f"replacement texts for candidate \"{' '.join(t.text for t in candidate)}\"!"
                )
                continue

            candidate_replacements[tuple(candidate)] = replacement_texts

        augmented_docs = []
        for _ in range(num_augments):
            changed = False
            augmented_doc = doc.copy(clear=[])
            possible_candidates = list(candidate_replacements.keys())

            chosen_candidates: typing.List[typing.Tuple[PetToken, ...]] = []
            for _ in range(self.replacements_per_document):
                if len(possible_candidates) == 0:
                    break
                chosen_candidate: typing.Tuple[PetToken, ...] = random.choice(
                    possible_candidates
                )
                chosen_candidates.append(chosen_candidate)
                possible_candidates.remove(chosen_candidate)
            chosen_candidates = sorted(
                chosen_candidates,
                key=lambda c: min(t.index_in_document for t in c),
                reverse=True,
            )

            for chosen_candidate in chosen_candidates:
                replacements = candidate_replacements[chosen_candidate]
                replacement = random.choice(replacements)
                replacements.remove(replacement)
                if len(candidate_replacements[chosen_candidate]) == 0:
                    del candidate_replacements[chosen_candidate]

                mutate.replace_sequence_inplace(
                    augmented_doc,
                    chosen_candidate[0].index_in_document,
                    chosen_candidate[-1].index_in_document + 1,
                    replacement,
                )
                changed = True
            num_annotations_after = (
                len(augmented_doc.mentions),
                len(augmented_doc.entities),
                len(augmented_doc.relations),
            )
            assert num_annotations_before == num_annotations_after, (
                f"Replacing candidates changed the documents annotations! "
                f"This must not happen! Before augmentation there were {num_annotations_before} "
                f"mentions, entities and relations, now there are {num_annotations_after}."
            )
            if changed:
                augmented_docs.append(augmented_doc)

        return augmented_docs

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        all_candidates = self.get_replacement_candidates(doc)
        random.shuffle(all_candidates)

        planned_replacements = [all_candidates]
        if len(all_candidates) > self.replacements_per_document:
            planned_replacements = itertools.combinations(
                all_candidates, r=self.replacements_per_document
            )

        docs = []
        for planned_replacement in planned_replacements:
            docs.extend(
                self.augment_single_doc(doc, planned_replacement, num_augments, 5)
            )
            if len(docs) >= num_augments:
                break

        return docs
