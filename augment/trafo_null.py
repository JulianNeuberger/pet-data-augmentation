import typing

from augment import base, params
from data import model


class TrafoNullStep(base.AugmentationStep):
    name = "null"

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    def do_augment(self, doc: model.Document) -> model.Document:
        return doc.copy()
