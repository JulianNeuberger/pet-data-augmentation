import json
import os.path
import pathlib
import typing

import numpy as np
import sklearn
from tqdm import tqdm

import augment
import data
import pipeline


def main():
    num_repeats = 10

    augmentation_rate = 3.0

    device = "CPU"
    device_ids = None

    # pipeline_step_class = pipeline.CrfMentionEstimatorStep
    # kwargs = {}

    pipeline_step_class = pipeline.CatBoostRelationExtractionStep
    kwargs = {
        "num_trees": 100,
        "device": device,
        "device_ids": device_ids,
    }

    stability_results: typing.Dict[str, typing.List[float]] = {}

    stats_path = (
        pathlib.Path(__file__).parent
        / "res"
        / "stability"
        / pipeline_step_class.__name__
    ).resolve()
    stats_path.mkdir(exist_ok=True)

    if os.path.isfile(stats_path / "runs.json"):
        with open(stats_path / "runs.json", "r") as f:
            stability_results = json.load(f)

    label: str
    aug_step: typing.Type[augment.AugmentationStep]
    # for label, aug_step in {"synonym": augment.SynonymSubstitution}.items():
    for label, aug_step in augment.all_augmentations.items():
        print(f"Running {label} ...")
        remaining_runs = num_repeats
        if label in stability_results:
            num_runs = len(stability_results[label])
            print(f"Already ran {num_runs} times ...")
            remaining_runs = num_repeats - num_runs
            if remaining_runs <= 0:
                print(f"Already done with {label}")
                continue
            print(f"Continuing with {label}, adding remaining {remaining_runs} runs")

        for _ in tqdm(range(remaining_runs)):
            data_path = (
                pathlib.Path(__file__).parent / "jsonl" / "all.new.jsonl"
            ).resolve()
            documents = data.pet.NewPetFormatImporter(str(data_path)).do_import()
            kf = sklearn.model_selection.KFold(
                n_splits=5, random_state=None, shuffle=True
            )
            fold_indices = kf.split(documents)

            aug_result = augment.augment_folds(
                documents,
                fold_indices,
                lambda train_docs: aug_step.get_default_configuration(train_docs),
                augmentation_rate,
            )

            unaugmented_pipeline_step = pipeline_step_class(
                name=pipeline_step_class.__name__, **kwargs
            )
            un_augmented_results = pipeline.cross_validate_pipeline(
                p=pipeline.Pipeline(
                    name=f"augmentation-{pipeline_step_class.__name__}",
                    steps=[unaugmented_pipeline_step],
                ),
                train_folds=aug_result.original_train_folds,
                test_folds=aug_result.test_folds,
                save_results=False,
            )
            un_augmented_f1 = un_augmented_results[
                unaugmented_pipeline_step
            ].overall_scores.f1

            augmented_pipeline_step = pipeline_step_class(
                name="crf mention extraction", **kwargs
            )
            augmented_results = pipeline.cross_validate_pipeline(
                p=pipeline.Pipeline(
                    name=f"augmentation-{pipeline_step_class.__name__}",
                    steps=[augmented_pipeline_step],
                ),
                train_folds=aug_result.augmented_train_folds,
                test_folds=aug_result.test_folds,
                save_results=False,
            )

            augmented_f1 = augmented_results[augmented_pipeline_step].overall_scores.f1
            improvement = augmented_f1 - un_augmented_f1

            if label not in stability_results:
                stability_results[label] = []
            stability_results[label].append(improvement)

            with open(stats_path / "runs.json", "w") as f:
                json.dump(stability_results, f)

    stats = {
        l: {
            "mean": np.mean(s),
            "std": np.std(s),
            "min": np.min(s),
            "max": np.max(s),
        }
        for l, s in stability_results.items()
    }
    stats = {
        k: v
        for k, v in sorted(
            stats.items(), key=lambda item: item[1]["mean"], reverse=True
        )
    }
    for l, s in stats.items():
        ps = ", ".join([f"{k}: {v:.4%}" for k, v in s.items()])
        print(f"{l} | {ps}")

    with open(stats_path / "stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    main()
