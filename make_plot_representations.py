from typing import Tuple

import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
from data import load_dataset
from util.mlflow.constants import ESM, EVE, ONE_HOT, TRANSFORMER, EVE_DENSITY
import seaborn as sns
from pathlib import Path

def plot_reduced_representations(dataset: str, representation: str, augmentation: str=None) -> None:
    # TODO: load data and observations
    # TODO: UMAP reduce X
    # TODO: figure of 2D X colored by assay values
    X, Y = load_dataset(dataset, representation=representation, augmentation=augmentation)
    reducer = UMAP(transform_seed=42)
    emb = reducer.fit_transform(X)
    plt.scatter(emb[:, 0], emb[:, 1], c=Y, cmap="magma", s=30, alpha=0.75, edgecolors="black")
    plt.title(f"2D UMAP {dataset} {representation}")
    # plt.savefig(f"./results/figures/representations/{dataset}_{representation}_UMAP.png")
    # plt.savefig(f"./results/figures/representations/{dataset}_{representation}_UMAP.pdf")
    plt.show()


def plot_reduced_representations_all_datasets(datasets: Tuple[str, ...], representations: Tuple[str, ...], augmentation=None):

    # Dictionary to map representation to name
    name_dict = {
        ONE_HOT: "One-Hot",
        EVE: "EVE",
        TRANSFORMER: "ProtBert",
        ESM: "ESM",
    }
    font = {'fontname': 'DejaVu Sans', 'fontsize': 40}

    fig, ax = plt.subplots(2, len(datasets) * 2, figsize=((len(datasets) + 2)*5, 8))

    for i, dataset in enumerate(datasets):
        umap_path = Path("results", "cache", f"{dataset}_representations.csv")

        # Load data / generate + save UMAP embeddings
        if not umap_path.exists():
            df = pd.DataFrame(columns=["representation", "x", "y", "target"])
            for representation in representations:
                X, Y = load_dataset(dataset, representation=representation, augmentation=augmentation)
                reducer = UMAP(transform_seed=42)
                emb = reducer.fit_transform(X)
                df_rep = pd.DataFrame({"representation": representation, "x": emb[:, 0], "y": emb[:, 1], "target": Y.squeeze()})
                df = pd.concat((df, df_rep), axis=0)
            df.to_csv(umap_path)
        else:
            df = pd.read_csv(umap_path, index_col=0)

        # Plot in 2x2 blocks
        ax_i = ax[:, (i*2):(i*2+2)]
        for j, representation in enumerate(representations):
            ax_ij = ax_i.flatten()[j]
            df_sub = df[df["representation"] == representation].sort_values(by="target", ascending=True)
            sns.scatterplot(
                data=df_sub,
                x="x",
                y="y",
                hue="target",
                ax=ax_ij,
                s=50,
                alpha=0.75,
                edgecolor="none",
                palette="magma",
                legend=False,
            )
            ax_ij.set_title(name_dict[representation], **font)
            # Clean up axes
            ax_ij.tick_params(
                bottom=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
                left=False,
            )
            ax_ij.set_xlabel("")
            ax_ij.set_ylabel("")
            ax_ij.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"./results/figures/representations/all_datasets_UMAP.pdf")
    plt.savefig(f"./results/figures/representations/all_datasets_UMAP.png")
    plt.show()


if __name__ == "__main__":
    datasets = ["1FQG", "UBQT", "TIMB", "MTH3", "BRCA"]
    # representations = [EVE_DENSITY, TRANSFORMER, ONE_HOT, ESM, EVE]
    representations = [ONE_HOT, EVE, TRANSFORMER, ESM]

    plot_reduced_representations_all_datasets(datasets, representations, augmentation=None)
