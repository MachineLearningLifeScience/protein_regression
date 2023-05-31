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
    dataset_key_map = {
        "1FQG": f"{chr(946)}-lactamase",
        "UBQT": "Ubiquitin",
        "TIMB": "TIM-Barrel",
        "MTH3": "MTH3",
        "BRCA": "BRCA",
    }
    font_kwargs = {'family': 'Arial', 'fontsize': 30, "weight": 'bold'}

    # Exact figure size might need tweaking
    fig, ax = plt.subplots(4, len(datasets), figsize=(20, 10))

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

        df["median"] = (df["target"] > df["target"].median()).astype(int)

        # Plot in 2x2 blocks
        for j, representation in enumerate(representations):
            ax_ij = ax[j, i]
            df_sub = df[df["representation"] == representation]
            sns.scatterplot(
                data=df_sub,
                x="x",
                y="y",
                hue="median",
                ax=ax_ij,
                s=50,
                alpha=0.75,
                edgecolor="none",
                palette=["#636EFA", "#FFA15A"],
                legend=False,
            )
            # Clean up axes
            ax_ij.tick_params(
                bottom=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
                left=False,
            )
            if i == 0:
                ax_ij.set_ylabel(name_dict[representation], **font_kwargs)
                ax_ij.yaxis.set_label_coords(-.275, .5, transform=ax_ij.transAxes)
            else:
                ax_ij.set_ylabel("")

            if j == 0:
                ax_ij.set_title(dataset_key_map[dataset], **font_kwargs)
            ax_ij.set_xlabel("")
            ax_ij.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, left=0.075, right=0.975)
    plt.savefig(f"./results/figures/representations/all_datasets_UMAP.pdf")
    plt.savefig(f"./results/figures/representations/all_datasets_UMAP.png")
    plt.show()


if __name__ == "__main__":
    datasets = ["1FQG", "UBQT", "TIMB", "MTH3", "BRCA"]
    # representations = [EVE_DENSITY, TRANSFORMER, ONE_HOT, ESM, EVE]
    representations = [ONE_HOT, EVE, TRANSFORMER, ESM]

    plot_reduced_representations_all_datasets(datasets, representations, augmentation=None)
