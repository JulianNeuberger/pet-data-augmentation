import dataclasses
import os
import time
import typing

import pandas as pd
import tqdm

import data.pet
from data import PetDocument
from eval import metrics
from pipeline import common, step

FoldStats = typing.List[typing.Dict[str, metrics.Stats]]


@dataclasses.dataclass
class PrintableScores:
    scores_by_tag: typing.Dict[str, metrics.Scores]
    overall_scores: metrics.Scores

    def __add__(self, other):
        scores_by_tag = {}
        for k in list(self.scores_by_tag.keys()) + list(other.scores_by_tag.keys()):
            if k not in self.scores_by_tag:
                scores_by_tag[k] = other.scores_by_tag[k]
            elif k not in other.scores_by_tag:
                scores_by_tag[k] = self.scores_by_tag[k]
            else:
                scores_by_tag[k] = self.scores_by_tag[k] + other.scores_by_tag[k]

        return PrintableScores(
            scores_by_tag=scores_by_tag,
            overall_scores=self.overall_scores + other.overall_scores,
        )

    def __truediv__(self, other):
        return PrintableScores(
            scores_by_tag={k: (v / other) for k, v in self.scores_by_tag.items()},
            overall_scores=self.overall_scores / other,
        )


def cross_validate_pipeline(
    p: common.Pipeline,
    *,
    train_folds: typing.List[typing.List[PetDocument]],
    test_folds: typing.List[typing.List[PetDocument]],
    save_results: bool = False,
    dump_predictions_dir: str = None,
    averaging_mode: str = "micro",
):
    assert len(train_folds) == len(test_folds)
    pipeline_results = []
    for n_fold, (train_fold, test_fold) in tqdm.tqdm(
        enumerate(zip(train_folds, test_folds)),
        total=len(train_folds),
        desc="cross validation fold",
    ):
        start = time.time_ns()
        ground_truth = [d.copy(clear=[]) for d in test_fold]
        print(
            f"copy of {len(test_fold)} documents took {(time.time_ns() - start) / 1e6:.4f}ms"
        )

        pipeline_result = p.run(
            train_documents=train_fold,
            test_documents=test_fold,
            ground_truth_documents=ground_truth,
        )
        pipeline_results.append(pipeline_result)
    res = accumulate_pipeline_results(pipeline_results, averaging_mode=averaging_mode)
    if dump_predictions_dir is not None:
        for i, pipeline_result in enumerate(pipeline_results):
            os.makedirs(dump_predictions_dir, exist_ok=True)
            out_path = os.path.join(dump_predictions_dir, f"fold-{i}.json")
            data.pet.PetJsonLinesExporter(out_path).export(
                pipeline_result.step_results[p.steps[-1]].predictions
            )

    if save_results:
        df_persistence = "experiments.pkl"
        if os.path.isfile(df_persistence):
            df: pd.DataFrame = pd.read_pickle(df_persistence)
        else:
            df = pd.DataFrame(
                columns=["experiment_name", "tag", "p", "r", "f1"]
            ).set_index(["experiment_name", "tag"])
        final_result = res[p.steps[-1]]

        new_rows = []
        for tag, value in final_result.scores_by_tag.items():
            new_rows.append(
                {
                    "experiment_name": p.name,
                    "tag": tag,
                    "p": value.p,
                    "r": value.r,
                    "f1": value.f1,
                }
            )
        new_rows.append(
            {
                "experiment_name": p.name,
                "tag": "overall",
                "p": final_result.overall_scores.p,
                "r": final_result.overall_scores.r,
                "f1": final_result.overall_scores.f1,
            }
        )

        new_rows_df = pd.DataFrame.from_records(new_rows).set_index(
            ["experiment_name", "tag"]
        )

        df = new_rows_df.combine_first(df)

        pd.to_pickle(df, df_persistence)

    print_pipeline_results(p, res)
    return res


def f1_stats_from_pipeline_result(
    result: common.PipelineResult, average_mode: str
) -> typing.Dict[step.PipelineStep, PrintableScores]:
    res: typing.Dict[step.PipelineStep, PrintableScores] = {}
    for pipeline_step, step_results in result.step_results.items():
        scores_by_ner = {
            k: metrics.Scores.from_stats(v) for k, v in step_results.stats.items()
        }
        if average_mode == "micro":
            combined_stats = sum(step_results.stats.values(), metrics.Stats(0, 0, 0))
            overall_scores = metrics.Scores.from_stats(combined_stats)
        elif average_mode == "macro":
            macro_scores = sum(scores_by_ner.values(), metrics.Scores(0, 0, 0)) / len(
                scores_by_ner
            )
            overall_scores = macro_scores
        else:
            raise ValueError(f"Unknown averaging mode {average_mode}.")
        res[pipeline_step] = PrintableScores(
            scores_by_tag=scores_by_ner, overall_scores=overall_scores
        )

    return res


def accumulate_pipeline_results(
    pipeline_results: typing.List[common.PipelineResult], averaging_mode: str = "micro"
) -> typing.Dict[step.PipelineStep, PrintableScores]:
    scores = []
    for pipeline_result in pipeline_results:
        scores.append(f1_stats_from_pipeline_result(pipeline_result, averaging_mode))

    num_results = len(scores)
    steps = pipeline_results[0].step_results.keys()

    return {
        step: sum(
            [s[step] for s in scores], PrintableScores({}, metrics.Scores(0, 0, 0))
        )
        / num_results
        for step in steps
    }


def print_pipeline_results(
    p: common.Pipeline, res: typing.Dict[step.PipelineStep, PrintableScores]
):
    print(f'=== {p.name} {"=" * (47 - len(p.name))}')
    for s, scores in res.items():
        print(f'--- {s.name} {"-" * (47 - len(s.name))}')
        print_scores(scores.scores_by_tag, scores.overall_scores)


def accumulate(
    left: typing.Dict[str, metrics.Stats], right: typing.Dict[str, metrics.Stats]
) -> typing.Dict[str, metrics.Stats]:
    key_set = set(left.keys()).union(set(right.keys()))
    return {
        ner_tag: left.get(ner_tag, metrics.Stats(1, 1, 1))
        + right.get(ner_tag, metrics.Stats(1, 1, 1))
        for ner_tag in key_set
    }


def print_scores(
    scores_by_ner: typing.Dict[str, metrics.Scores],
    overall_score: metrics.Scores,
    order: typing.List[str] = None,
):
    len_ner_tags = max([len(t) for t in scores_by_ner.keys()])

    if order is None:
        order = list(scores_by_ner.keys())

    print(f'{" " * (len_ner_tags - 2)}Tag |   P     |   R     |   F1    ')
    print(f'{"=" * (len_ner_tags + 2)}+=========+=========+========')

    for ner_tag in order:
        if ner_tag not in scores_by_ner:
            continue
        score = scores_by_ner[ner_tag]
        print(
            f" {ner_tag: >{len_ner_tags}} "
            f"| {score.p: >7.2%} "
            f"| {score.r: >7.2%} "
            f"| {score.f1: >7.2%}"
        )
    print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')

    print(
        f' {"Overall": >{len_ner_tags}} '
        f"| {overall_score.p: >7.2%} "
        f"| {overall_score.r: >7.2%} "
        f"| {overall_score.f1: >7.2%}"
    )
    print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')
