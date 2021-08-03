import numpy as np
import matplotlib.pyplot as plt


def plot_metric_for_dataset(datasets: list, metric_values: list, reps: list, cvtype: str):
    c = ['darkred', 'dimgray', 'blue', 'darkorange', 'k', 'green', 'purple']
    plt.figure(figsize=(15,10))
    for i, res in enumerate(metric_values):
        seps = np.linspace(-0.1, 0.1, len(res))
        for j, r in enumerate(res):
            mse = np.mean(r)
            std = np.std(r, ddof=1)
            plt.errorbar(i+seps[j], mse, yerr = std, fmt='o', capsize=4, capthick=2, color=c[j], label=reps[j])
    plt.title('Accuracy of protein regression methods '+ cvtype, size=20)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    plt.legend(markers, reps, bbox_to_anchor=(1, 1), numpoints=1, prop={'size':16})
    plt.xticks(list(range(len(datasets))), datasets, size=16)
    plt.yticks(size=16)
    plt.xlabel('Protein data set', size=20)
    plt.ylabel('MSE', size=20)
    plt.tight_layout()
    plt.savefig('results/figures/'+'accuracy of methods '+cvtype)
    plt.show()

