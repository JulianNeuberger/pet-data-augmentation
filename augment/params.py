import abc
import dataclasses
import functools
import itertools
import numbers
import operator
import typing


@dataclasses.dataclass
class Param(abc.ABC):
    name: str


NT = typing.TypeVar("NT", bound=numbers.Number)


@dataclasses.dataclass
class NumberParam(typing.Generic[NT], Param):
    min_value: typing.Optional[NT]
    max_value: typing.Optional[NT]


class IntegerParam(NumberParam[int]):
    pass


class FloatParam(NumberParam[float]):
    pass


CT = typing.TypeVar("CT")


@dataclasses.dataclass
class ChoiceParam(typing.Generic[CT], Param):
    choices: typing.List[CT]
    max_num_picks: int = 1

    def get_combinations_as_bit_masks(self):
        combinations = []
        indices = list(range(len(self.choices)))
        for length in range(1, self.max_num_picks + 1):
            for combination in itertools.combinations(indices, length):
                bit_mask: int = functools.reduce(operator.ior, combination)
                combinations.append(bit_mask)
        return combinations

    def bit_mask_to_choices(self, bit_mask: int) -> typing.List[CT]:
        return [self.choices[i] for i in self.bits(bit_mask)]

    @staticmethod
    def bits(number):
        bit = 1
        while number >= bit:
            if number & bit:
                yield bit
            bit <<= 1


@dataclasses.dataclass
class BooleanParameter(Param):
    pass
