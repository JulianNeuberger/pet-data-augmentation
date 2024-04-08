import typing

from augment import base, params
from data import model


class CheatingTransformationStep(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], test_dataset: typing.List[model.Document]):
        self._test_data = test_dataset
        self.index = 0
        super().__init__(dataset)

    def do_augment(self, doc: model.Document) -> model.Document:
        self.index += 1
        return self._test_data[(self.index - 1) % len(self._test_data)].copy()

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []
