import os
import pathlib
import typing

import numpy as np
import seaborn as sns
import sklearn
import tqdm
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from openTSNE import TSNEEmbedding, TSNE
from openai import OpenAI

import augment
import data

client = OpenAI(
    base_url="http://132.180.195.1:8007/v1",
    api_key="ollama",  # required, but unused
)

sns.set_theme(rc={"axes.grid": False})


def get_embedding(text: typing.Union[typing.List[str], str], model):
    if type(text) is str:
        text = [text.replace("\n", " ")]
    text = [t.replace("\n", " ") for t in text]
    return [d.embedding for d in client.embeddings.create(input=text, model=model).data]


def embed_documents(
    documents: typing.List[data.PetDocument],
    key: str,
    model: str,
    strategy: typing.Literal[
        "sentence", "document", "mention", "relation"
    ] = "document",
    force_recalculation: bool = False,
) -> np.ndarray:
    embedding_dir = (
        pathlib.Path(__file__).parent / "res" / "embeddings" / model / strategy
    )
    os.makedirs(embedding_dir, exist_ok=True)

    save_file_path = str(embedding_dir / f"{key}.npy")

    if os.path.exists(save_file_path) and not force_recalculation:
        print("Reusing pre-calculated embeddings!")
        return np.load(save_file_path)

    vectors = []
    for d in tqdm.tqdm(documents, desc=f"{strategy} embedding"):
        texts = []
        if strategy == "document":
            texts = [" ".join(t.text for t in d.tokens)]
        elif strategy == "sentence":
            texts = [" ".join(t.text for t in s) for s in d.sentences]
        elif strategy == "mention":
            texts = [m.text(d) for m in d.mentions]
        elif strategy == "relation":
            texts = [
                (
                    f"{d.mentions[r.head_mention_index].text(d)} "
                    f"{r.type} "
                    f"{d.mentions[r.tail_mention_index].text(d)}"
                )
                for r in d.relations
            ]
        embedded_texts = get_embedding(texts, model)
        vectors.extend(embedded_texts)

    nd_array = np.array(vectors)
    np.save(save_file_path, nd_array)
    return nd_array


def plot_data(
    title: str,
    ax: plt.Axes,
    *,
    original_tsne: np.ndarray,
    augmented_tsne: typing.Dict[str, np.ndarray],
    marker_size: float = 5,
    palette: typing.List = None,
    markers: typing.List[str] = None,
):
    if palette is None:
        palette = sns.color_palette()
    if markers is None:
        markers = ["o", "v", "X", "^", "*", "<", "s", ">", "D"]

    legend_handles: typing.List[typing.Tuple[PathCollection, str]] = []

    original_color = palette.pop(0)
    original_marker = markers.pop(0)

    for i, (label, v) in enumerate(augmented_tsne.items()):
        handle = ax.scatter(
            v[:, 0],
            v[:, 1],
            s=marker_size,
            color=palette.pop(0),
            marker=markers.pop(0),
            alpha=0.5,
        )
        legend_handles.append((handle, label))

    original_handle = ax.scatter(
        original_tsne[:, 0],
        original_tsne[:, 1],
        s=marker_size,
        color=original_color,
        marker=original_marker,
        alpha=1,
    )
    ax.set_title(title)
    legend_handles.append((original_handle, "Original Data"))

    return legend_handles


def run_augmentation(
    originals: typing.List[data.PetDocument],
    steps: typing.List[typing.Type[augment.AugmentationStep]],
    augmentation_rate: float,
    key: str,
    force_update: bool = False,
) -> typing.List[data.PetDocument]:
    save_dir = pathlib.Path(__file__).parent / "res" / "augmentations"
    os.makedirs(save_dir, exist_ok=True)
    save_file_path = str(save_dir / f"{key}.jsonl")

    if os.path.isfile(save_file_path) and not force_update:
        print(f"Reusing pre-calculated augmentations for {key}!")
        return data.NewPetFormatImporter(save_file_path).do_import()

    print(f"Calculating augmentations for {key}")
    augmented_docs = augment.run_augmentation(
        originals,
        [s.get_default_configuration(originals) for s in steps],
        augmentation_rate,
    )
    data.PetJsonLinesExporter(save_file_path).export(augmented_docs)
    return augmented_docs


def main():
    data_path = (pathlib.Path(__file__).parent / "jsonl" / "all.new.jsonl").resolve()
    originals = data.pet.NewPetFormatImporter(str(data_path)).do_import()
    kf = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    fold_indices = list(kf.split(originals))
    train_indices, _ = fold_indices[0]
    train_documents = [originals[i] for i in train_indices]

    # model = "bge-m3"
    # model = "nomic-embed-text"
    model = "jina/jina-embeddings-v2-base-en"

    embedding_strategy: typing.Literal[
        "document", "sentence", "mention", "relation"
    ] = "relation"

    original_vectors = embed_documents(
        documents=originals, model=model, key="originals", strategy=embedding_strategy
    )

    print("Fitting TSNE reduction for original vectors ...")
    tsne_transformer: TSNEEmbedding = TSNE(
        perplexity=10,
        early_exaggeration="auto",
        random_state=42,
        # n_jobs=12,
        metric="cosine",
    ).fit(original_vectors)
    original_embedded = tsne_transformer.transform(original_vectors)
    print("Done!")

    fig: plt.Figure
    ax: plt.Axes
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8, 8),
        sharey="all",
        sharex="all",
        gridspec_kw={"height_ratios": [0.5, 4, 4]},
    )
    axes = axes.flat

    legend_axes = axes[:2]
    plot_axes = axes[2:]

    main_color = sns.color_palette()[0]
    other_color = sns.color_palette()[1]

    legend_handles = None

    for ax, (category, pipelines) in zip(
        plot_axes, augment.augmentation_classes.items()
    ):
        augmented_embedded_vectors = {}
        for i, (label, steps) in enumerate(pipelines.items()):
            key = label.replace(" ", "_")
            augmented_docs = run_augmentation(
                originals, steps, 2, key=key, force_update=False
            )
            augmented_vectors = embed_documents(
                documents=augmented_docs,
                model=model,
                key=key,
                strategy=embedding_strategy,
            )
            print(f"Running TSNE reduction for {key} ...")
            augmented_embedded = tsne_transformer.transform(augmented_vectors)
            augmented_embedded_vectors[label] = augmented_embedded
            print("Done!")

        print(f"Plotting {category}")
        num_augmentations = len(pipelines)
        legend_handles = plot_data(
            title=category,
            ax=ax,
            original_tsne=original_embedded,
            augmented_tsne=augmented_embedded_vectors,
            marker_size=15,
            markers=["*"] + ["o"] * num_augmentations,
            palette=[main_color] + [other_color] * num_augmentations,
        )
    assert legend_axes is not None
    assert len(legend_handles) >= 2
    originals_handle = legend_handles[-1]
    augmented_handle = legend_handles[0]
    legend_axes[-1].legend(
        [originals_handle[0], augmented_handle[0]],
        ["original", "augmented"],
        ncol=2,
    )

    for ax in legend_axes:
        ax.set_axis_off()

    fig.tight_layout()

    # plt.legend([h for h, l in legend_handles], [l for h, l in legend_handles])
    save_dir = pathlib.Path(__file__).parent.absolute() / "figures" / "feature-space"
    plt.savefig(save_dir / f"{embedding_strategy}.png", dpi=900)
    plt.savefig(save_dir / f"{embedding_strategy}.pdf")


if __name__ == "__main__":
    main()
