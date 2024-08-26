import typing

from augment import base, params
from data import model


class CheatingTransformationStep(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        test_dataset: typing.List[model.Document],
    ):
        self._test_data = test_dataset
        self.index = 0
        super().__init__(dataset)

    def do_augment(
        self, doc: model.Document, num_augments: int
    ) -> typing.List[model.Document]:
        ret = []
        for _ in range(num_augments):
            ret.append(self._test_data[self.index].copy())
            self.index += 1
            self.index %= len(self._test_data)
        return ret

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []
