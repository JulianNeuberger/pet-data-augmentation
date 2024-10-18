import dataclasses
import inspect
import random
import typing

import data
from augment.abbreviate import (
    ContractionsAndExpansionsPerturbation,
    InsertAbbreviations,
    ReplaceAbbreviationsAndAcronyms,
    UseAcronyms,
)
from augment.antonyms import EvenAntonymsSubstitute, AntonymInversionStep
from augment.augmenter import Augmenter, run_augmentation
from augment.base import AugmentationStep
from augment.deletion import RandomDeletion
from augment.inserting import RandomInsert, FillerWordAugmentation
from augment.llm import LargeLanguageModelRephrasing
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
from augment.repeat import (
    InverseMentionTypeFrequencySampler,
    InverseRelationTypeFrequencySampler,
    UniformRepeat,
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


@dataclasses.dataclass
class AugmentedFolds:
    test_folds: typing.List[typing.List[data.PetDocument]]
    original_train_folds: typing.List[typing.List[data.PetDocument]]
    augmented_train_folds: typing.List[typing.List[data.PetDocument]]


def augment_folds(
    documents: typing.List[data.PetDocument],
    fold_indices: typing.Iterable[
        typing.Tuple[typing.Iterable[int], typing.Iterable[int]]
    ],
    step_factory: typing.Callable[[typing.List[data.PetDocument]], AugmentationStep],
    augmentation_rate: float,
) -> AugmentedFolds:
    un_augmented_train_folds = []
    augmented_train_folds: typing.List[typing.List[data.PetDocument]] = []
    test_folds = []
    for train_indices, test_indices in fold_indices:
        test_documents = [documents[i] for i in test_indices]
        un_augmented_train_documents = [documents[i] for i in train_indices]

        step = step_factory(un_augmented_train_documents)

        augmented_train_documents = run_augmentation(
            un_augmented_train_documents, [step], augmentation_rate
        )
        random.shuffle(augmented_train_documents)
        augmented_train_folds.append(augmented_train_documents)
        un_augmented_train_folds.append(un_augmented_train_documents)
        print(
            f"Augmented {len(un_augmented_train_documents)} documents "
            f"with augmentation rate of {augmentation_rate:.4f} "
            f"resulting in {len(augmented_train_documents)} documents"
        )
        test_folds.append(test_documents)
    return AugmentedFolds(
        original_train_folds=un_augmented_train_folds,
        augmented_train_folds=augmented_train_folds,
        test_folds=test_folds,
    )


########################################################
# REPHRASING AUGMENTATIONS                             #
########################################################
rephrasing_augmentations = {
    "Hypernym Replacement": HypernymReplacement,
    "Hyponym Replacement": HyponymReplacement,
    "Synonym Substitution": SynonymSubstitution,
    "Back-Translation": BackTranslation,
    "Multi-Lingual Back-Translation": MultiLingualBackTranslation,
    "Transformer Fill": TransformerFill,
    "Even Antonym Substitute": EvenAntonymsSubstitute,
}

########################################################
# REPEATING AUGMENTATIONS                              #
########################################################
repeating_augmentations = {
    "Merge Documents": MergeDocumentsStep,
    "Mention Replacement": EntityMentionReplacement,
    "Subsequence Substitution": TagSubsequenceSubstitution,
    "Uniform Oversampling": UniformRepeat,
    "Inverse Frequency Oversampling": InverseMentionTypeFrequencySampler,
}

########################################################
# NOISE AUGMENTATIONS                                  #
########################################################
noising_augmentations = {
    "Random Deletion": RandomDeletion,
    "Filler Words": FillerWordAugmentation,
    "Random Insert": RandomInsert,
    "Antonym Switch": AntonymInversionStep,
    "Auxiliary Negation Removal": AuxiliaryNegationRemoval,
    "Synonym Insertion": SynonymInsertion,
}

########################################################
# REORDERING AUGMENTATIONS                             #
########################################################
reordering_augmentations = {
    "Sentence Reordering": SentenceReordering,
    "Shuffle Within Segments": ShuffleWithinSegments,
    "Random Swap": RandomTokenSwap,
}

all_augmentations = {}
all_augmentations.update(rephrasing_augmentations)
all_augmentations.update(repeating_augmentations)
all_augmentations.update(noising_augmentations)
all_augmentations.update(reordering_augmentations)

augmentation_classes = {
    "Repeating": repeating_augmentations,
    "Reordering": reordering_augmentations,
    "Rephrasing": rephrasing_augmentations,
    "Adding Noise": noising_augmentations,
}
