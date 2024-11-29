import json
import os
import pathlib
import time

import augment
import data

data_path = (pathlib.Path(__file__).parent / "jsonl" / "all.new.jsonl").resolve()
all_documents = data.pet.NewPetFormatImporter(str(data_path)).do_import()
docs_by_id = {d.id: d for d in all_documents}

examples_path = (pathlib.Path(__file__).parent / "res" / "examples").resolve()
os.makedirs(examples_path, exist_ok=True)

example_document = docs_by_id["doc-3.3"]
# example_document = docs_by_id["doc-10.2"]
all_documents.remove(example_document)
dataset = all_documents

augmentation_steps = [
    # augment.FillerWordAugmentation.get_default_configuration(dataset,)
    # augment.RandomDeletion(dataset=dataset, p=0.15)
    # augment.UseAcronyms.get_default_configuration(dataset=dataset)
    # augment.SentenceReordering(dataset=dataset)
    # augment.UniformRepeat(dataset=dataset),
    # augment.SynonymSubstitution(dataset=dataset, replace_probability=0.3),
    # augment.RandomDeletion(dataset=dataset, p=0.1),
    # augment.HypernymReplacement(dataset=dataset, replace_probability=0.4),
    # augment.BackTranslation.get_default_configuration(dataset),
    # augment.TransformerFill.get_default_configuration(dataset)
    # augment.ContextualMeaningPerturbation.get_default_configuration(dataset),
    # augment.ShuffleWithinSegments.get_default_configuration(dataset),
    # augment.EntityMentionReplacement.get_default_configuration(dataset),
    augment.LostInTranslation.get_default_configuration(dataset)
]

augmented = augment.run_augmentation(
    [example_document], augmentation_steps, augmentation_rate=10
)

examples_name = str(augmentation_steps[0].__class__.__name__)
if len(augmentation_steps) > 1:
    examples_name += f"_plus_{len(augmentation_steps) - 1}"
target_file = examples_path / f"{examples_name}.jsonl"
data.pet.PetJsonLinesExporter(str(target_file)).export(augmented)
