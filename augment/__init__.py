import inspect
import typing

from augment.augmenter import Augmenter, run_augmentation

from augment.base import AugmentationStep

from augment.abbreviate import (
    ContractionsAndExpansionsPerturbation,
    InsertAbbreviations,
    ReplaceAbbreviationsAndAcronyms,
)
from augment.antonyms import EvenAntonymsSubstitute, AntonymInversionStep
from augment.deletion import RandomDeletion
from augment.inserting import RandomInsert, FillerWordAugmentation
from augment.masking import (
    TransformerFill,
    HyponymReplacement,
    ContextualMeaningPerturbation,
    HypernymReplacement,
)
from augment.merge import MergeDocumentsStep
from augment.negation import AuxiliaryNegationRemoval
from augment.reordering import (
    SentenceReordering,
    RandomTokenSwap,
    ShuffleWithinSegments,
)
from augment.sampling import TagSubsequenceSubstitution, EntityMentionReplacement
from augment.synonyms import SynonymInsertion, SynonymSubstitution
from augment.translation import (
    MultiLingualBackTranslation,
    BackTranslation,
    LostInTranslation,
)


def collect_all_augmentations(
    base_class: typing.Type,
) -> typing.List[typing.Type[base.AugmentationStep]]:
    sub_classes = []

    immediate_sub_classes = base_class.__subclasses__()
    sub_classes.extend([c for c in immediate_sub_classes if not inspect.isabstract(c)])
    for sub_class in immediate_sub_classes:
        child_sub_classes = collect_all_augmentations(sub_class)
        for child_sub_class in child_sub_classes:
            sub_classes.append(child_sub_class)

    return sub_classes
