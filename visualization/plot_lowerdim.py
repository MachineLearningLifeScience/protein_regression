from curses.ascii import SP
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
import matplotlib.pyplot as plt
from pyparsing import alphas
from util.mlflow.convenience_functions import get_mlflow_results, check_results
from util.mlflow.constants import MSE, SPEARMAN_RHO, LINEAR, NON_LINEAR, VAE
from typing import List
from tqdm import tqdm
from scipy.optimize import curve_fit

def f(x, a, b, c):
    return a * np.log(b*x) + c

def plot_lower_dim_results(datasets: List[str], algorithms: List[str], representations: List[str], cv_types: List[str],
                            dimensions: List[int]=[2, 10, 100, 1000, None], metrics: List[str]=[MSE, SPEARMAN_RHO], dim_reduction=LINEAR, VAE_DIM=30):
    dim_results = {metric: {d: {} for d in dimensions} for metric in metrics}
    cv_names = [cv(datasets[0]).get_name() for cv in cv_types]
    ds = [d for d in dimensions.copy()[:-1]] + [10000]
    for metric in tqdm(metrics):
        for dim in tqdm(dimensions):
            for split in tqdm(cv_types, leave=False):
                # TODO what happens if multiple metrics are queried?
                results = get_mlflow_results(datasets=datasets, algos=algorithms, reps=representations, metrics=[metric], 
                                            train_test_splitter=split, dim=dim, dim_reduction=dim_reduction)
                dim_results[metric][dim][split(datasets[0]).get_name()] = results
    
    # make multiplot row per representation, column per split
    fig, ax = plt.subplots(len(cv_types), len(representations)*2, figsize=(20, 10))
    colors = ['darkred', 'green', 'blue', 'hotpink', 'teal']
    xx = np.linspace(0, 1000)
    for i, cv in enumerate(cv_names):
        for data in datasets:
            j_idx = 0
            for rep in representations:
                _ds = ds if rep != VAE else [d for d in ds if d < VAE_DIM] + [1050]
                for k, method in enumerate(dim_results[metrics[0]][dimensions[0]][cv][data].keys()):
                    # Note: currently only plots one-element datasets list, otherwise too large/convuluted plot
                    first_metric = [dim_results[metrics[0]][d][cv][data][method][rep][None][metrics[0]] for d in dimensions]
                    first_metric, correct = check_results(first_metric)
                    average_metric = np.average(first_metric, axis=1) if correct else np.nanmean(first_metric, axis=1)
                    one_minus_mean_nmses = np.ones(len(dimensions)) - average_metric
                    std_nmses = np.nanstd(first_metric, axis=1)
                    second_metric = [dim_results[metrics[1]][d][cv][datasets[0]][method][rep][None][metrics[1]] for d in dimensions]
                    second_metric, correct = check_results(second_metric)
                    mean_rhos = np.average(second_metric, axis=1) if correct else np.nanmean(second_metric, axis=1)
                    std_rhos = np.nanstd(second_metric, axis=1)
                    for r, mse, std_r, std_mse, _d in zip(mean_rhos, one_minus_mean_nmses, std_rhos, std_nmses, _ds):
                        ax[i, j_idx].errorbar(_d, mse, yerr=std_mse, fmt='D', c=colors[k], ms=2, mew=2, label=f"{method}", alpha=(0.5+int(correct)*0.5)) # shade if values are missing by alpha
                        ax[i, j_idx+1].errorbar(_d, r, yerr=std_r, fmt="o", c=colors[k], ms=2, mew=2, label=f"{method}", alpha=(0.5+int(correct)*0.5))
                    # fit regression line
                    _ds = np.array(_ds) # VAE arrays, repeat last dim-observations for length of array, but len(_ds) needs to be len(observations)
                    popt_mse, _pcov = curve_fit(f, _ds, np.nan_to_num(one_minus_mean_nmses))
                    popt_r, _pcov = curve_fit(f, _ds, np.nan_to_num(mean_rhos))
                    # mse_coeff = poly.fit(_ds, one_minus_mean_nmses[:len(_ds)], deg=2, domain=[0, 1.e+04], window=[0, 1])
                    # r_coeff = poly.fit(_ds, mean_rhos[:len(_ds)], deg=2, domain=[0, 1.e+04], window=[0,1])
                    # ax[i, j_idx].plot(xx, mse_coeff(xx), alpha=0.6)
                    # ax[i, j_idx+1].plot(xx, r_coeff(xx), alpha=0.6)
                    ax[i, j_idx].plot(xx, f(xx, *popt_mse), "k--", alpha=0.6)
                    ax[i, j_idx+1].plot(xx, f(xx, *popt_r), "k--", alpha=0.6)
                ax[i, j_idx].set_xticks(_ds)
                ax[i, j_idx].set_xscale('log')
                ax[i, j_idx+1].set_xscale('log')
                ax[i, j_idx].set_yticks(np.arange(-0.25, 1.1, 0.25))
                ax[i, j_idx+1].set_yticks(np.arange(-0.25, 1.1, 0.25))
                ax[i, j_idx].tick_params(axis='both', which='major', labelsize=9)
                ax[i, j_idx].tick_params(axis='both', which='minor', labelsize=5)
                ax[i, j_idx].grid(True, color='k', alpha=0.25)
                ax[i, j_idx+1].tick_params(axis='both', which='major', labelsize=9)
                ax[i, j_idx+1].tick_params(axis='both', which='minor', labelsize=5)
                ax[i, j_idx+1].grid(True, color='k', alpha=0.25)
                ax[i, j_idx+1].set_title(f"{rep}", fontsize=9)
                ax[i, j_idx+1].set_ylabel(f"{cv}\n 1-NMSE", rotation=90, fontsize=10)
                ax[i, j_idx+1].set_ylabel(f"{metrics[1]}", rotation=90, fontsize=10)
                ax[i, j_idx].set_xlabel("Dimensions")
                j_idx += 2
    ax[i, j_idx-1].legend(prop={'size': 6})#bbox_to_anchor=(1.1, 1.05))
    plt.suptitle(f"Regression in lower dimensions {dim_reduction} \n on {str(datasets)}")
    plt.tight_layout()
    plt.savefig(f"./results/figures/dim_reduction/lower_dim{dimensions}_{datasets}_{representations}_{metrics}_{dim_reduction}", bbox_inches='tight')
    plt.show()

