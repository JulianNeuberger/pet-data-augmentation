import json
import os
import pathlib
import random
import sys
import traceback
import typing

import optuna
import sklearn.model_selection

import augment
import data
import pipeline
from augment import params
from data import PetDocument

strategies: typing.List[typing.Type[augment.AugmentationStep]] = [
    augment.SynonymSubstitution
]

max_runs_per_step = 25
random_state = 42
random.seed(random_state)


def suggest_param(param: params.Param, trial: optuna.Trial) -> typing.Any:
    if isinstance(param, params.IntegerParam):
        return trial.suggest_int(
            name=param.name, low=param.min_value, high=param.max_value
        )

    if isinstance(param, params.FloatParam):
        return trial.suggest_float(
            name=param.name, low=param.min_value, high=param.max_value
        )

    if isinstance(param, params.ChoiceParam):
        if param.max_num_picks > 1:
            choices = param.get_combinations_as_bit_masks()
            choice = trial.suggest_categorical(name=param.name, choices=choices)
            choice_as_list = param.bit_mask_to_choices(choice)
            return choice_as_list
        return trial.suggest_categorical(name=param.name, choices=param.choices)

    if isinstance(param, params.BooleanParameter):
        return trial.suggest_categorical(name=param.name, choices=[True, False])


def instantiate_step(
    step_class: typing.Type[augment.AugmentationStep],
    trial: optuna.Trial,
    dataset: typing.List[PetDocument],
) -> augment.AugmentationStep:
    suggested_params = {
        p.name: suggest_param(p, trial) for p in step_class.get_params()
    }
    return step_class(dataset, **suggested_params)


def objective_factory(
    augmenter_class: typing.Type[augment.AugmentationStep],
    pipeline_step_class: typing.Type[pipeline.PipelineStep],
    documents: typing.List[PetDocument],
    fold_indices: typing.List[typing.Tuple[typing.Iterable[int], typing.Iterable[int]]],
    un_augmented_f1: float,
    **kwargs,
):
    def objective(trial: optuna.Trial):

        augmentation_rate = trial.suggest_float("augmentation_rate", low=0.0, high=10.0)
        un_augmented_train_folds = []
        augmented_train_folds: typing.List[typing.List[PetDocument]] = []
        test_folds = []
        for train_indices, test_indices in fold_indices:
            test_documents = [documents[i] for i in test_indices]
            un_augmented_train_documents = [documents[i] for i in train_indices]

            step = instantiate_step(
                augmenter_class, trial, un_augmented_train_documents
            )

            augmented_train_documents = augment.run_augmentation(
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

        augmented_pipeline_step = pipeline_step_class(
            name="crf mention extraction", **kwargs
        )
        augmented_results = pipeline.cross_validate_pipeline(
            p=pipeline.Pipeline(
                name=f"augmentation-{pipeline_step_class.__name__}",
                steps=[augmented_pipeline_step],
            ),
            train_folds=augmented_train_folds,
            test_folds=test_folds,
            save_results=False,
        )

        augmented_f1 = augmented_results[augmented_pipeline_step].overall_scores.f1
        improvement = augmented_f1 - un_augmented_f1
        print(f"Improvement of {improvement:.2%}")
        return improvement

    return objective


def main():
    device = "CPU"
    device_ids = None

    if len(sys.argv) > 1:
        assert len(sys.argv) == 3, (
            "If you specify devices to train on, please specify either GPU/CPU "
            "as first argument and the id(s) as second one, see "
            "https://catboost.ai/en/docs/features/training-on-gpu"
        )
        device = sys.argv[1]
        device_ids = sys.argv[2]

    for strategy_class in strategies:
        errors = []
        try:
            strategy_class.validate_params(strategy_class)
        except TypeError as e:
            errors.append(e)
        if len(errors) > 0:
            raise AssertionError("\n".join([str(e) for e in errors]))

    # all_documents = loader.read_documents_from_json("./jsonl/all.jsonl")
    all_documents = data.pet.NewPetFormatImporter(r"jsonl\all.new.jsonl").do_import()
    kf = sklearn.model_selection.KFold(
        n_splits=5, random_state=random_state, shuffle=True
    )
    fold_indices = list(kf.split(all_documents))

    # pipeline_step_class = pipeline.CrfMentionEstimatorStep
    pipeline_step_class = pipeline.CatBoostRelationExtractionStep

    kwargs = {
        "num_trees": 100,
        "device": device,
        "device_ids": device_ids,
    }

    train_folds: typing.List[typing.List[PetDocument]] = []
    test_folds: typing.List[typing.List[PetDocument]] = []
    for train_indices, test_indices in fold_indices:
        test_documents = [all_documents[i] for i in test_indices]
        train_documents = [all_documents[i] for i in train_indices]
        train_folds.append(train_documents)
        test_folds.append(test_documents)

    unaugmented_pipeline_step = pipeline_step_class(
        name="crf mention extraction", **kwargs
    )

    run_info_path = (
        pathlib.Path(__file__)
        .parent.joinpath("res")
        .joinpath("runs")
        .joinpath("info.json")
        .resolve()
    )
    os.makedirs(pathlib.Path(run_info_path).parent.resolve(), exist_ok=True)

    un_augmented_f1: typing.Optional[float] = None
    if os.path.exists(run_info_path):
        with open(run_info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)
            if pipeline_step_class.__name__ in run_info:
                un_augmented_f1 = run_info[pipeline_step_class.__name__]
    else:
        run_info = {}

    if un_augmented_f1 is None:
        un_augmented_results = pipeline.cross_validate_pipeline(
            p=pipeline.Pipeline(
                name=f"augmentation-{pipeline_step_class.__name__}",
                steps=[unaugmented_pipeline_step],
            ),
            train_folds=train_folds,
            test_folds=test_folds,
            save_results=False,
        )
        un_augmented_f1 = un_augmented_results[
            unaugmented_pipeline_step
        ].overall_scores.f1
        run_info[pipeline_step_class.__name__] = un_augmented_f1
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f)

    for strategy_class in strategies:
        print(f"Running optimization for strategy {strategy_class.__name__}")
        objective = objective_factory(
            strategy_class,
            pipeline_step_class,
            all_documents,
            fold_indices,
            un_augmented_f1,
            **kwargs,
        )
        study = optuna.create_study(
            direction="maximize",
            load_if_exists=True,
            study_name=f"{strategy_class.__name__}-{pipeline_step_class.__name__}",
            # storage="mysql://optuna@localhost/pet_data_augment",
        )
        trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if len(trials) >= max_runs_per_step:
            print(
                f"All trials for {strategy_class.__name__} already ran, continuing..."
            )
            continue
        try:
            study.optimize(
                objective,
                callbacks=[
                    optuna.study.MaxTrialsCallback(
                        n_trials=max_runs_per_step,
                        states=(optuna.trial.TrialState.COMPLETE,),
                    )
                ],
            )
        except Exception as e:
            if type(e) == KeyboardInterrupt:
                raise e
            print(f"Error in strategy {strategy_class.__name__}, skipping.")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
