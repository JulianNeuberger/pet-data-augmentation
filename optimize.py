import time

import numpy as np
import optuna

import data
import main
import pipeline


def runner():
    def re_extraction(trial: optuna.Trial):
        start = time.time()

        params = {
            'num_trees': trial.suggest_int(name='num_trees', low=10, high=5_000),
            'negative_sampling_rate': trial.suggest_float(name='negative_sampling_rate', low=0.0, high=50.0),
            'context_size': trial.suggest_int(name='context_size', low=0, high=10),
            'depth': trial.suggest_int(name='depth', low=1, high=12),
            'use_pos_features': trial.suggest_categorical(name='use_pos_features', choices=[True, False]),
            'use_embedding_features': trial.suggest_categorical(name='use_embedding_features', choices=[True, False]),
        }

        print(f'Using params')
        print(params)

        res = main.cross_validate_pipeline(
            p=pipeline.Pipeline(name=f'hyper parameter optimization trial #{trial.number}', steps=[
                pipeline.CatBoostRelationExtractionStep(name=f'trial #{trial.number}', **params)
            ]),
            train_folds=train_folds,
            test_folds=test_folds
        )
        objective_value = list(res.values())[0].overall_scores.f1
        print(f'Ran for {time.time() - start:.4f}s and produced {objective_value:.2%} F1')
        return objective_value

    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    study = optuna.create_study(study_name='catboost-optimization',
                                storage='sqlite:///optuna.db',
                                direction='maximize',
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler())
    study.optimize(re_extraction)


if __name__ == '__main__':
    runner()
