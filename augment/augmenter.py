import copy
import typing
from random import shuffle

import tqdm

from augment import base
from data import model

# Author: Leonie
def run_augmentation_old(dataset: typing.List[model.Document], step: base.AugmentationStep) -> typing.List[model.Document]:
    return [step.do_augment(doc) for doc in dataset]


# Author: Benedikt
def run_augmentation(dataset: typing.List[model.Document], step: base.AugmentationStep, aug_rate):
    num_of_doc_to_aug = int(len(dataset) * aug_rate)
    print(f'Augmenting {len(dataset)} documents with '
          f'augmentation factor of {aug_rate:.4f} '
          f'using strategy {step.__class__.__name__}...')
    extended_dataset = []
    ds = copy.deepcopy(dataset)
    indices = []
    for i in range(len(dataset)):
        indices.append(i)
    shuffle(indices)
    for i in range(num_of_doc_to_aug):
        if i < len(ds):
            index = indices[i]
            extended_dataset.append(ds[index])
        else:
            x = i % len(ds)
            index = indices[x]
            extended_dataset.append(ds[index])
    unaug_dataset = copy.deepcopy(dataset)
    ext_ds = copy.deepcopy(extended_dataset)
    unaug_dataset.extend(ext_ds)
    aug_dataset = []

    for doc in tqdm.tqdm(extended_dataset):
        aug_dataset.append(step.do_augment(doc))

    #aug_dataset = [step.do_augment(doc) for doc in extended_dataset]
    aug_data = copy.deepcopy(dataset)
    aug_data.extend(aug_dataset)
    indices2 = []
    for i in range(len(unaug_dataset)):
        indices2.append(i)
    shuffle(indices2)
    unaug_dataset_shuff = []
    aug_dataset_shuff = []
    for i in range(len(indices2)):
        unaug_dataset_shuff.append(unaug_dataset[indices2[i]])
        aug_dataset_shuff.append(aug_data[indices2[i]])
    return aug_dataset_shuff, unaug_dataset_shuff
