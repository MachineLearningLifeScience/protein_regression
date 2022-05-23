from curses.ascii import SP
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
from util.mlflow.convenience_functions import get_mlflow_results
from util.mlflow.constants import MSE, SPEARMAN_RHO, LINEAR, NON_LINEAR
from typing import List

def plot_lower_dim_results(datasets: List[str], algorithms: List[str], representations: List[str], cv_types: List[str],
                            dimensions: List[int]=[2, 10, 100, 1000, None], metrics: List[str]=[MSE, SPEARMAN_RHO], dim_reduction=LINEAR):
    dim_results = {metric: {d: {} for d in dimensions} for metric in metrics}
    cv_names = [cv(datasets[0]).get_name() for cv in cv_types]
    ds = [d for d in dimensions.copy()[:-1]] + [10000]
    for metric in metrics:
        for dim in dimensions:
            for split in cv_types:
                # TODO what happens if multiple metrics are queried?
                results = get_mlflow_results(datasets=datasets, algos=algorithms, reps=representations, metrics=[metric], 
                                            train_test_splitter=split, dim=dim, dim_reduction=dim_reduction)
                dim_results[metric][dim][split(datasets[0]).get_name()] = results
    
    # make multiplot row per representation, column per split
    fig, ax = plt.subplots(len(cv_types), len(representations)*2, figsize=(20, 10))
    colors = ['darkred', 'green', 'blue', 'hotpink', 'teal']
    for i, cv in enumerate(cv_names):
        j_idx = 0
        for rep in representations: 
            for k, method in enumerate(dim_results[MSE][dimensions[0]][cv][datasets[0]].keys()):
                # Note: currently only plots one-element datasets list, otherwise too large/convuluted plot
                nmses = [dim_results[MSE][d][cv][datasets[0]][method][rep][None][MSE] for d in dimensions]
                one_minus_mean_nmses = np.ones(len(dimensions)) - np.average(nmses, axis=1)
                std_nmses = np.std(nmses, axis=1)
                rhos = [dim_results[SPEARMAN_RHO][d][cv][datasets[0]][method][rep][None][SPEARMAN_RHO] for d in dimensions]
                mean_rhos = np.average(rhos, axis=1)
                std_rhos = np.std(rhos, axis=1)
                for r, mse, std_r, std_mse, _d in zip(mean_rhos, one_minus_mean_nmses, std_rhos, std_nmses, ds):
                    ax[i, j_idx].errorbar(_d, mse, yerr=std_mse, fmt='D', c=colors[k], ms=2, mew=2, label=f"{method}")
                    ax[i, j_idx+1].errorbar(_d, r, yerr=std_r, fmt="o", c=colors[k], ms=2, mew=2, label=f"{method}")
            ax[i, j_idx].set_xticks(ds)
            ax[i, j_idx].set_xscale('log')
            ax[i, j_idx+1].set_xscale('log')
            ax[i, j_idx].set_yticks(np.arange(0, 1.1, 0.25))
            ax[i, j_idx+1].set_yticks(np.arange(-1, 1.1, 0.25))
            ax[i, j_idx].tick_params(axis='both', which='major', labelsize=9)
            ax[i, j_idx].tick_params(axis='both', which='minor', labelsize=5)
            ax[i, j_idx].grid(True, color='k', alpha=0.25)
            ax[i, j_idx].set_title(f"{rep}", fontsize=9)
            ax[i, j_idx].set_ylabel(f"{cv}\n 1-NMSE", rotation=90, fontsize=10)
            ax[i, j_idx+1].set_ylabel(f"Spearman r", rotation=90, fontsize=10)
            ax[i, j_idx].set_xlabel("Dimensions")
            j_idx += 2
    ax[i, j_idx-1].legend(prop={'size': 6})#bbox_to_anchor=(1.1, 1.05))
    plt.suptitle(f"Regression in lower dimensions {dim_reduction} \n on {datasets[0]}")
    plt.tight_layout()
    plt.savefig(f"./results/figures/lower_dim{dimensions}_{datasets[0]}_{representations}_{metrics}_{dim_reduction}", bbox_inches='tight')
    plt.show()

