import abc
import inspect
import random
import typing

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
        last_mention: typing.Optional[PetMention] = doc.get_mention_for_token(
            doc.tokens[0]
        )
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
        self, dataset: typing.List[PetDocument], replace_probability: float, **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.p = replace_probability

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [params.FloatParam("replace_probability", min_value=0.0, max_value=1.0)]

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

    def should_replace(
        self, candidate: typing.List[PetToken], num_current_replacements: int
    ) -> bool:
        return random.random() < self.p

    def augment_single_doc(
        self,
        doc: PetDocument,
        candidates: typing.List[typing.List[PetToken]],
        num_augments: int,
    ) -> typing.List[PetDocument]:
        print()
        print()
        print(f"------ {doc.name} -------------------------------------------")

        num_annotations_before = (
            len(doc.mentions),
            len(doc.entities),
            len(doc.relations),
        )

        if len(candidates) == 0:
            print(
                f'Replacement strategy "{self.__class__.__name__}" did not find any candidates!'
            )

        random.shuffle(candidates)

        candidate_replacements = {}
        replacement_plans: typing.List[
            typing.List[typing.Tuple[typing.List[PetToken], typing.List[str]]]
        ] = []
        for i in range(num_augments):
            print(f"-- {i} ---------------")
            replacement_plans.append([])
            cur_num_augments = 0
            for candidate in candidates:
                if not self.should_replace(candidate, cur_num_augments):
                    continue
                candidate_text = " ".join(t.text for t in candidate)
                if candidate_text not in candidate_replacements:
                    replacements = self.get_replacements(candidate, 5, doc)
                    candidate_replacements[candidate_text] = replacements
                if len(candidate_replacements[candidate_text]) == 0:
                    continue
                replacement = random.choice(candidate_replacements[candidate_text])
                replacement_plans[-1].append((candidate, replacement))
                cur_num_augments += 1

        augmented_docs = []
        for plan in replacement_plans:
            changed = False
            augmented_doc = doc.copy(clear=[])

            plan = sorted(
                plan,
                key=lambda x: min(t.index_in_document for t in x[0]),
                reverse=True,
            )

            for c, r in plan:
                mutate.replace_sequence_inplace(
                    augmented_doc,
                    c[0].index_in_document,
                    c[-1].index_in_document + 1,
                    r,
                )
                changed = True
                print(f"replace \"{[t.text for t in c]}\" with \"{r}\"")

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

        # # candidate_replacements: typing.Dict[
        # #     typing.Tuple[PetToken, ...], typing.Dict[str, typing.List[typing.List[str]]]
        # # ] = {}
        # for candidate in candidates:
        #     replacement_texts = self.get_replacements(candidate, 5, doc)
        #     # remove identical replacements
        #     replacement_texts = [
        #         t
        #         for t in replacement_texts
        #         if " ".join(c.text for c in candidate) != " ".join(r for r in t)
        #     ]
        #     replacement_texts = [t for t in replacement_texts if len(t) > 0]
        #     if len(replacement_texts) == 0:
        #         print(
        #             f'Replacement strategy "{self.__class__.__name__}" did not generate any '
        #             f"replacement texts for candidate \"{' '.join(t.text for t in candidate)}\"!"
        #         )
        #         continue
        #
        #     candidate_replacements[tuple(candidate)] = {
        #         "unused": replacement_texts,
        #         "all": list(replacement_texts),
        #     }
        #
        # augmented_docs = []
        # for _ in range(num_augments):
        #     changed = False
        #     augmented_doc = doc.copy(clear=[])
        #     candidates = list(candidate_replacements.keys())
        #     candidates = [
        #         c for i, c in enumerate(candidates) if self.should_replace(list(c), i)
        #     ]
        #     candidates = sorted(
        #         candidates,
        #         key=lambda c: min(t.index_in_document for t in c),
        #         reverse=True,
        #     )
        #
        #     num_replacements = 0
        #     for candidate in candidates:
        #         # try to pick an unused replacement and fall back to used ones
        #         replacements = candidate_replacements[candidate]["unused"]
        #         is_unused = True
        #         if len(replacements) == 0:
        #             replacements = candidate_replacements[candidate]["all"]
        #             is_unused = False
        #
        #         replacement: typing.List[str] = random.choice(replacements)
        #         if is_unused:
        #             replacements.remove(replacement)
        #
        #         if len(replacement) == 0:
        #             print("Zero length replacement, should be filtered.")
        #             continue
        #
        #         mutate.replace_sequence_inplace(
        #             augmented_doc,
        #             candidate[0].index_in_document,
        #             candidate[-1].index_in_document + 1,
        #             replacement,
        #         )
        #         num_replacements += 1
        #         changed = True
        #     num_annotations_after = (
        #         len(augmented_doc.mentions),
        #         len(augmented_doc.entities),
        #         len(augmented_doc.relations),
        #     )
        #     assert num_annotations_before == num_annotations_after, (
        #         f"Replacing candidates changed the documents annotations! "
        #         f"This must not happen! Before augmentation there were {num_annotations_before} "
        #         f"mentions, entities and relations, now there are {num_annotations_after}."
        #     )
        #     if changed:
        #         augmented_docs.append(augmented_doc)

        return augmented_docs

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        all_candidates = self.get_replacement_candidates(doc)
        return self.augment_single_doc(doc, all_candidates, num_augments)
