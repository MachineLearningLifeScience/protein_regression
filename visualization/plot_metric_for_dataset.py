import numpy as np
import matplotlib.pyplot as plt


def plot_metric_for_dataset(datasets: list, metric_values: np.ndarray, reps: list, cvtype: str):
    import matplotlib.pyplot as plt
    c = ['darkred', 'darkgray', 'blue', 'darkorange', 'k', 'green', 'purple']
    plt.figure(figsize=(15,10))
    for i, res in enumerate(metric_values):
        mse = res[0]
        std = res[1]
        for j,e in enumerate(mse):
            plt.errorbar(i, e/mse[0], yerr = std[j], fmt='.', capsize=2, color=c[j], label=reps[j])
        if i == 0 and cvtype=='bio':
            val = 0.9089174928692626
            err = 0.0845948576415664
            plt.errorbar(i, val/mse[0], yerr = err, fmt='.', capsize=2, color=c[-1], label=reps[-1])
    plt.title('Accuracy of protein regression methods', size=20)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    plt.legend(markers, reps, bbox_to_anchor=(1, 1), numpoints=1)
    plt.xticks(list(range(len(datasets))), datasets, size=16)
    plt.xlabel('Datasets', size=20)
    plt.ylabel('MSE/(MSE onehot)', size=20)
    plt.tight_layout()
    if cvtype == 'bio':
        subscript = 'in bio-relevant cv'
    if cvtype == 'reg':
        subscript = 'in regular cv'
    plt.savefig('results/figures/'+'accuracy of methods '+subscript)
    plt.show()

