from umap import UMAP
import matplotlib.pyplot as plt
from data import load_dataset
from util.mlflow.constants import ESM, EVE, ONE_HOT, TRANSFORMER, EVE_DENSITY


def plot_reduced_representations(dataset: str, representation: str, augmentation: str=None) -> None:
    # TODO: load data and observations
    # TODO: UMAP reduce X
    # TODO: figure of 2D X colored by assay values
    X, Y = load_dataset(dataset, representation=representation, augmentation=augmentation)
    reducer = UMAP(transform_seed=42)
    emb = reducer.fit_transform(X)
    plt.scatter(emb[:, 0], emb[:, 1], c=Y, cmap="magma", s=30, alpha=0.75, edgecolors="black")
    plt.title(f"2D UMAP {dataset} {representation}")
    plt.savefig(f"./results/figures/representations/{dataset}_{representation}_UMAP.png")
    plt.savefig(f"./results/figures/representations/{dataset}_{representation}_UMAP.pdf")
    #plt.show()


if __name__ == "__main__":
    datasets = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA"] # ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA"]
    representations = [EVE_DENSITY, TRANSFORMER, ONE_HOT, ESM, EVE]
    for dataset in datasets:
        for representation in representations:
            plot_reduced_representations(dataset, representation)
