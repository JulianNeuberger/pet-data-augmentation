import typing

from augment import base, params
from data import model


class Trafo110Step(base.AbbreviationStep):
    def __init__(self, dataset: typing.List[model.Document]):
        abbreviations = self._load()
        super().__init__(dataset, abbreviations)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    @staticmethod
    def _load() -> typing.Dict[str, str]:
        abbreviations = {}
        with open(f"./resources/abbreviations/110.tsv", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                key, value = line.split("\t", 1)
                abbreviations[key] = value
        return abbreviations
