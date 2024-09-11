import abc
import random
import typing

from augment import base, params
from data import PetDocument


class UniformRepeat(base.AugmentationStep):
    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        return [doc.copy(clear=[]) for _ in range(num_augments)]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []


class InverseTypeFrequencySampler(base.AugmentationStep, abc.ABC):
    def __init__(self, dataset: typing.List[PetDocument], **kwargs):
        super().__init__(dataset, **kwargs)
        self.documents_by_type: typing.Dict[str, typing.List[PetDocument]] = (
            self.get_documents_by_type()
        )
        occurrences: typing.Dict[str, int] = self.count_type_occurrences()
        total = sum(occurrences.values())
        inverse_frequencies = {t: total - o for t, o in occurrences.items()}
        self.document_weights: typing.Dict[str, int] = {d.name: 0 for d in self.dataset}
        for t, v in inverse_frequencies.items():
            for d in self.documents_by_type[t]:
                self.document_weights[d.name] += v
        self.documents_by_id = {d.name: d for d in self.dataset}

    def count_type_occurrences(self) -> typing.Dict[str, int]:
        return {t: len(ds) for t, ds in self.get_documents_by_type().items()}

    @abc.abstractmethod
    def get_documents_by_type(self) -> typing.Dict[str, typing.List[PetDocument]]:
        raise NotImplementedError()

    def do_augment(
        self, doc: PetDocument, num_augments: int
    ) -> typing.List[PetDocument]:
        documents = []
        weights = []

        for d, w in self.document_weights.items():
            documents.append(d)
            weights.append(w)

        document_ids = random.choices(documents, weights=weights, k=num_augments)
        return [self.documents_by_id[i].copy(clear=[]) for i in document_ids]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []


class InverseMentionTypeFrequencySampler(InverseTypeFrequencySampler):
    def get_documents_by_type(self) -> typing.Dict[str, typing.List[PetDocument]]:
        ret = {}
        for d in self.dataset:
            for m in d.mentions:
                if m.type not in ret:
                    ret[m.type] = []

                ret[m.type].append(d)
        return ret


class InverseRelationTypeFrequencySampler(InverseTypeFrequencySampler):
    def get_documents_by_type(self) -> typing.Dict[str, typing.List[PetDocument]]:
        ret = {}
        for d in self.dataset:
            for m in d.relations:
                if m.type not in ret:
                    ret[m.type] = []

                ret[m.type].append(d)
        return ret
