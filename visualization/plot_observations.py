import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_observations(y: np.ndarray, x_line: float, name: str):
    sns.kdeplot(np.ravel(-y), color="darkblue", fill=True)
    plt.vlines(0., 0, 0.7, colors=["red"], linestyles="dashed", linewidths=2.5)
    plt.savefig(f"./results/figures/data_dist/{name}_dist.png")
    plt.savefig(f"./results/figures/data_dist/{name}_dist.pdf")
    plt.show()

