import numpy as np
import matplotlib.pyplot as plt


def plot_metric_for_dataset(datasets: list, metric_values: list, reps: list, cvtype: str):
    c = ['darkred', 'dimgray', 'blue', 'darkorange', 'k', 'green', 'purple']
    plt.figure(figsize=(15,10))
    for i, res in enumerate(metric_values):
        for j, r in enumerate(res):
            mse = np.mean(r)
            std = np.std(r, ddof=1)
            plt.errorbar(i, mse, yerr = std, fmt='o', capsize=4, color=c[j], label=reps[j])
    plt.title('Accuracy of protein regression methods'+ cvtype, size=20)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    plt.legend(markers, reps, bbox_to_anchor=(1, 1), numpoints=1)
    plt.xticks(list(range(len(datasets))), datasets, size=16)
    plt.xlabel('Datasets', size=20)
    plt.ylabel('MSE/(MSE onehot)', size=20)
    plt.tight_layout()
    plt.savefig('results/figures/'+'accuracy of methods '+cvtype)
    plt.show()

