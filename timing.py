import json
import pathlib
import time

import augment
import data

data_path = (pathlib.Path(__file__).parent / "jsonl" / "all.new.jsonl").resolve()
all_documents = data.pet.NewPetFormatImporter(str(data_path)).do_import()
docs_by_id = {d.id: d for d in all_documents}

example_document = docs_by_id["doc-3.3"]
# example_document = docs_by_id["doc-10.2"]
all_documents.remove(example_document)
dataset = all_documents

all_augmentation_steps = augment.collect_all_augmentations(augment.AugmentationStep)

times_path = pathlib.Path(__file__).parent / "res" / "times.json"
times = {}
if times_path.is_file():
    with open(times_path, "r") as f:
        times = json.load(f)
for augment_step in all_augmentation_steps:
    if augment_step.__name__ in times:
        print(f"Skipping {augment_step.__name__} because it has already been processed")
        continue
    start = time.time()
    augmented = augment.run_augmentation(
        [example_document], [augment_step.get_default_configuration(dataset)], augmentation_rate=10
    )
    end = time.time()
    duration_seconds = end - start
    times[augment_step.__name__] = duration_seconds

    print(f"{augment_step.__name__}: {duration_seconds * 1000:.2E}ms")

    with open(times_path, "w", encoding="utf-8") as f:
        json.dump(times, f)

print("--------------------------------------")
print("\n")
for s, t in times.items():
    print(f"{s}: {t * 1000:.2E}ms ({t * 1000:.0f}ms, {t:.4f}s) ")
