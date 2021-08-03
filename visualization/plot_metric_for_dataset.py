import numpy as np
import matplotlib.pyplot as plt

def plot_metric_for_dataset(metric_values: dict, cvtype: str):
    c = ['darkred', 'dimgray', 'blue', 'darkorange', 'k', 'green', 'purple']
    plt.figure(figsize=(15,10))
    reps = []
    for i, dataset_key in enumerate(metric_values.keys()):
        num_exps = len(metric_values[dataset_key].keys())
        seps = np.linspace(-0.1, 0.1, num_exps)
        for j, rep_key in enumerate(metric_values[dataset_key].keys()):
            mse_list = metric_values[dataset_key][rep_key]
            mse = np.mean(mse_list)
            std = np.std(mse_list, ddof=1)
            plt.errorbar(i+seps[j], mse, yerr = std, fmt='o', capsize=4, capthick=2, color=c[j], label=rep_key)
    plt.title('Accuracy of protein regression methods '+ cvtype, size=20)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    plt.legend(markers, reps, bbox_to_anchor=(1, 1), numpoints=1, prop={'size':16})
    plt.xticks(list(range(len(metric_values.keys()))), metric_values.keys(), size=16)
    plt.yticks(size=16)
    plt.xlabel('Protein data set', size=20)
    plt.ylabel('MSE', size=20)
    plt.tight_layout()
    plt.savefig('results/figures/'+'accuracy of methods '+cvtype)
    plt.show()

