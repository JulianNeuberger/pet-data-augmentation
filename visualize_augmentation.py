import itertools
import os
import typing

import optuna
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt, ticker

import augment
import data
import pipeline
from augment import params
from data import loader, writer

sns.set_theme()
plt.rcParams["font.family"] = "CMU Serif"

transformation_classes: typing.List[typing.Type[augment.AugmentationStep]] = [
    augment.Trafo3Step,
    augment.Trafo5Step,
    augment.Trafo6Step,
    augment.Trafo8Step,  # long runtime
    augment.Trafo24Step,
    augment.Trafo26Step,
    augment.Trafo39Step,
    augment.Trafo40Step,
    augment.Trafo58Step,  # runs too long?
    augment.Trafo62Step,  # runs too long
    augment.Trafo79Step,
    augment.Trafo82Step,
    augment.Trafo86HyponymReplacement,
    augment.Trafo86HypernymReplacement,
    augment.Trafo88Step,
    augment.Trafo90Step,
    augment.Trafo100Step,
    augment.Trafo101Step,
    augment.Trafo103Step,
    augment.Trafo106Step,
    augment.TrafoInsertStep,
]


def get_vocab_size(documents: typing.List[data.Document]) -> int:
    vocab: typing.Set[str] = set()
    for document in documents:
        vocab = vocab.union(set([t.text for t in document.tokens]))
    return len(vocab)


def get_mean_span_length(documents: typing.List[data.Document]) -> float:
    span_lens = []

    for document in documents:
        for mention in document.mentions:
            span_lens.append(len(mention.token_indices))

    return sum(span_lens) / len(span_lens)


def get_relation_directions(
    documents: typing.List[data.Document],
) -> typing.Tuple[int, int]:
    num_ltr = 0
    num_rtl = 0
    for document in documents:
        for relation in document.relations:
            head_entity = document.entities[relation.head_entity_index]
            tail_entity = document.entities[relation.tail_entity_index]

            head_mentions = [document.mentions[i] for i in head_entity.mention_indices]
            head_mentions = [
                m for m in head_mentions if m.sentence_index in relation.evidence
            ]

            tail_mentions = [document.mentions[i] for i in tail_entity.mention_indices]
            tail_mentions = [
                m for m in tail_mentions if m.sentence_index in relation.evidence
            ]

            h: data.Mention
            t: data.Mention
            for h, t in itertools.product(head_mentions, tail_mentions):
                if (
                    h.document_level_token_indices(document)[0]
                    < t.document_level_token_indices(document)[0]
                ):
                    num_ltr += 1
                else:
                    num_rtl += 1
    return num_ltr, num_rtl


def add_labels(ax: plt.Axes, df: pd.DataFrame):
    texts = []
    xs = []
    ys = []
    df = df.reset_index()
    for _, row in df.iterrows():
        x = row["vocab_size"]
        y = row["span_length"]
        text = ax.text(x, y, row["name"])
        texts.append(text)
        xs.append(x)
        ys.append(y)
    return texts, xs, ys


def augment_data(transformation_class, step_class, original_data):
    try:
        study = optuna.load_study(
            study_name=f"{transformation_class.__name__}-{step_class.__name__}",
            storage="mysql://optuna@localhost/pet_data_augment",
        )
        best_trial = study.best_trial
        best_params = best_trial.params
    except ValueError:
        print(f"No config for {transformation_class.__name__}, skipping visualization.")
        return None
    param_definitions = {p.name: p for p in transformation_class.get_params()}

    transformation_kwargs = {}

    for param_name, best_value in best_params.items():
        if param_name not in param_definitions:
            continue
        param_definition = param_definitions[param_name]
        if isinstance(param_definition, params.ChoiceParam):
            if param_definition.max_num_picks > 1:
                best_value = param_definition.bit_mask_to_choices(best_value)
            elif best_value not in param_definition.choices:
                best_value = param_definition.choices[best_value]
        transformation_kwargs[param_name] = best_value

    print(f"Running {transformation_class.__name__} with args {transformation_kwargs}")

    transformation = transformation_class(
        dataset=original_data, **transformation_kwargs
    )

    augmentation_rate = best_params["augmentation_rate"]

    augmented_data, _ = augment.run_augmentation(
        original_data, transformation, augmentation_rate
    )

    return augmented_data


def build_plot_data():
    original_data = loader.read_documents_from_json("./jsonl/all.jsonl")
    original_vocab_size = get_vocab_size(original_data)
    original_span_length = get_mean_span_length(original_data)
    original_num_samples = len(original_data)
    original_num_ltr, original_num_rtl = get_relation_directions(original_data)

    plot_data = [
        {
            "name": "Original",
            "num_samples": original_num_samples,
            "vocab_size": original_vocab_size,
            "span_length": original_span_length,
            "num_ltr_relations": original_num_ltr,
            "num_rtl_relations": original_num_rtl,
            "type": "original",
        }
    ]

    step_classes = [
        pipeline.CatBoostRelationExtractionStep,
        pipeline.CrfMentionEstimatorStep,
    ]

    for step_class in step_classes:
        for transformation_class in transformation_classes:
            name = transformation_class.__name__
            name = name.replace("Trafo", "").replace("Step", "")
            variation = ""
            if name.endswith("HyponymReplacement"):
                name.replace("HyponymReplacement", "")
                variation = "Hyponym Replacement"

            if name.endswith("HypernymReplacement"):
                name.replace("HypernymReplacement", "")
                variation = "Hypernym Replacement"

            try:
                int(name)
                name = f"B.{name} {variation}"
            except ValueError:
                pass

            loaded_trafos = [p["name"] for p in plot_data]
            if name in loaded_trafos:
                print('Already loaded, continuing')
                continue

            base_dir = os.path.join("jsonl", "augmented")
            data_file = f"{os.path.join(base_dir, transformation_class.__name__)}.jsonl"

            if os.path.exists(data_file):
                print(f"Loading existing data for {transformation_class.__name__}")
                augmented_data = loader.read_documents_from_json(data_file)
            else:
                print(f"Creating new data for {transformation_class.__name__}")
                augmented_data = augment_data(
                    transformation_class, step_class, original_data
                )
                if augmented_data is None:
                    continue
                os.makedirs(base_dir, exist_ok=True)
                with open(data_file, "w") as f:
                    for d in augmented_data:
                        as_json = writer.dump_document_to_json(d)
                        f.write(f"{as_json}\n")

            vocab_size = get_vocab_size(augmented_data)
            span_length = get_mean_span_length(augmented_data)
            num_ltr, num_rtl = get_relation_directions(augmented_data)

            plot_data.append(
                {
                    "name": name,
                    "num_samples": len(augmented_data),
                    "vocab_size": vocab_size,
                    "span_length": span_length,
                    "type": "step",
                    "num_ltr_relations": num_ltr,
                    "num_rtl_relations": num_rtl,
                }
            )

    df = pd.DataFrame.from_records(plot_data)
    return df


def augmentation_effect_figure(df: pd.DataFrame):
    df = df.set_index("name")

    print(df)
    original_data = df.loc[["Original"]]
    print(original_data)

    print('------------------')
    print(df["span_length"])
    print('------------------')
    print(original_data["span_length"])
    print('------------------')
    df["vocab_size"] = (df["vocab_size"] - original_data["vocab_size"].item()) / original_data["vocab_size"].item()
    df["span_length"] = (df["span_length"] - original_data["span_length"].item()) / original_data["span_length"].item()

    sns.scatterplot(df, x="vocab_size", y="span_length", hue="type")
    ax = plt.gca()

    ax.get_legend().remove()

    ax.set_xlim(-0.05)
    ax.set_ylim(-0.05)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.title("transformation effects", fontname="CMU Serif")

    plt.ylabel("average mention length change", fontname="CMU Serif")
    plt.xlabel("vocabulary size change", fontname="CMU Serif")
    texts, xs, ys = add_labels(ax, df)

    adjust_text(
        texts=texts, x=xs, y=ys, arrowprops=dict(arrowstyle="-", color="k", lw=0.5)
    )

    plt.tight_layout()
    plt.savefig("figures/data-augmentation/trafo-effects.png")
    plt.savefig("figures/data-augmentation/trafo-effects.pdf")


def data_characteristics_figure(df: pd.DataFrame):
    total_num_relations = df["num_ltr_relations"] + df["num_rtl_relations"]
    df["ratio_ltr_relations"] = df["num_ltr_relations"] / total_num_relations
    df["ratio_rtl_relations"] = df["num_rtl_relations"] / total_num_relations
    df = df.drop(
        columns=[
            "vocab_size",
            "span_length",
            "type",
            "num_ltr_relations",
            "num_rtl_relations",
            "num_samples",
        ]
    )
    df = df.set_index("name")
    df.plot(kind="bar", stacked=True)
    ax: plt.Axes = plt.gca()
    ax.set_ylim(0.4, 1.05)
    threshold = df.loc[["Original"]]["ratio_ltr_relations"].item()
    plt.plot(ax.get_xlim(), [threshold, threshold])
    plt.legend(["original distribution", "right to left", "left to right"])
    plt.ylabel("share of relations")
    plt.xlabel("data augmentation technique")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # df["relation_direction_ratio"] = df["num_ltr_relations"] / df["num_rtl_relations"]
    # sns.barplot(df, x="name", y="relation_direction_ratio", hue="type")
    plt.tight_layout()
    plt.savefig("figures/data-augmentation/directionality.png")
    plt.savefig("figures/data-augmentation/directionality.pdf")


if __name__ == "__main__":
    def main():
        plot_data = build_plot_data()
        augmentation_effect_figure(plot_data.copy())
        data_characteristics_figure(plot_data.copy())
    main()
