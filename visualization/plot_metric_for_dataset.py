from distutils.log import error
from sys import stderr
from warnings import warn
from typing import List
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from data.load_dataset import load_dataset
from visualization.plot_metric_for_uncertainties import prep_reliability_diagram
from visualization import algorithm_colors as ac
from visualization import algorithm_markers as am
from visualization import augmentation_colors as aug_c
from visualization import representation_colors as rc
from visualization import representation_markers as rm
from visualization import task_colors as tc
from util.mlflow.constants import GP_L_VAR, LINEAR, VAE, EVE, VAE_DENSITY, ONE_HOT, EVE_DENSITY
from util.mlflow.constants import MLL, MSE, SPEARMAN_RHO, PAC_BAYES_EPS, STD_Y, PAC_BAYES_EPS
from util.postprocess import filter_functional_variant_data_less_than
from util.postprocess import filter_functional_variant_data_greater_than

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.family": "sans-serif",
})


def plot_metric_for_dataset(metric_values: dict, cvtype: str, dim):
    plt.figure(figsize=(15,10))
    reps = []
    for i, dataset_key in enumerate(metric_values.keys()):
        num_exps = len(metric_values[dataset_key].keys())
        seps = np.linspace(-0.2, 0.2, num_exps)
        for j, rep_key in enumerate(metric_values[dataset_key].keys()):
            if rep_key not in reps:
                reps.append(rep_key)
            mse_list = metric_values[dataset_key][rep_key]
            mse = np.mean(mse_list)
            std = np.std(mse_list, ddof=1)/np.sqrt(len(mse))
            plt.errorbar(i+seps[j], mse, yerr = std, fmt='o', capsize=4, capthick=2, color=rc.get(rep_key), label=rep_key)
    plt.title(f'Accuracy of regression methods using {cvtype} on d={dim}', size=20)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in rc.values()]
    plt.legend(markers, reps, bbox_to_anchor=(1, 1), numpoints=1, prop={'size':16})
    plt.xticks(list(range(len(metric_values.keys()))), metric_values.keys(), size=16)
    plt.yticks(size=16)
    plt.xlabel('Protein data set', size=20)
    plt.ylabel('MSE', size=20)
    plt.tight_layout()
    plt.savefig('results/figures/'+f'accuracy_of_methods_d={dim}_cv_{cvtype}.png')
    plt.savefig('results/figures/'+f'accuracy_of_methods_d={dim}_cv_{cvtype}.pdf')
    plt.show()


def barplot_metric_comparison(metric_values: dict, cvtype: str, metric: str, height=0.08):
    plot_heading = f'Comparison of algoritms and representations, cv-type: {cvtype} \n scaled, GP optimized zero-mean, var=0.4 (InvGamma(3,3)), len=0.1 (InvGamma(3,3)), noise=0.1 ∈ [0.01, 1.0] (Uniform)'
    filename = 'results/figures/benchmark/'+'accuracy_of_methods_barplot_'+cvtype+str(list(metric_values.keys()))
    fig, ax = plt.subplots(1, len(metric_values.keys()), figsize=(len(metric_values.keys())*4,10))
    axs = np.ravel(ax)
    reps = []
    for d, dataset_key in enumerate(metric_values.keys()):
        for i, algo in enumerate(metric_values[dataset_key].keys()):
            seps = np.linspace(-height*0.8*len(metric_values[dataset_key].keys()), 
                               height*0.8*len(metric_values[dataset_key].keys()), 
                               len(metric_values[dataset_key][algo].keys()))
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                if rep not in reps:
                    reps.append(rep)
                mse_list = metric_values[dataset_key][algo][rep][None][metric]
                neg_invert_mse = 1-np.mean(mse_list)
                error_on_mean = np.std(mse_list, ddof=1)/np.sqrt(len(mse_list))
                axs[d].errorbar(neg_invert_mse, i+seps[j], xerr=error_on_mean, color=rc.get(rep), ecolor="black",
                                marker=rm.get(rep), fillstyle='none', markersize=14, lw=5, capsize=6, label=rep)
            axs[d].axhline(i+seps[0]-0.2, -1, 1, c='grey', ls='-', alpha=0.75)
        axs[d].axvline(0, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
        axs[d].axvline(-1, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
        axs[d].axvline(0.75, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].axvline(0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(0.25, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].axvline(-0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(-0.25, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].axvline(-0.75, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].set_yticks(list(range(len(list(metric_values[dataset_key].keys())))))
        axs[d].set_yticklabels(['' for i in range(len(list(metric_values[dataset_key].keys())))])
        axs[0].set_yticklabels(list(metric_values[dataset_key].keys()), size=25, rotation=90)
        axs[d].set_xlim((-1, 1))
        axs[d].tick_params(axis='x', which='both', labelsize=22)
        axs[d].set_title(dataset_key, size=25)
        axs[d].set_xlabel('1-NMSE', size=25)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:len(reps)], labels[:len(reps)], loc='lower right', ncol=len(reps), prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def barplot_metric_comparison_bar(metric_values: dict, cvtype: str, metric: str, width: float=0.17, color_by: str="algo", x_axis: str="rep", augmentation=None, suffix=None) -> None:
    """
    Barplot Fig. 2 for comparisons (i.e. representation, algorithms, etc.)
    """
    if color_by.lower() not in ["algo", "rep", "task"]:
        warn("Misspecified color-scheme. Defaulting to color by algorithm.")
        color_by = "algo"
    if x_axis.lower() not in ["algo", "rep", "task"]:
        warn("Misspecified the plotting groups. Default to representations.")
        x_axis = "rep"
    splitters = list(metric_values.keys())
    datasets = list(metric_values[splitters[0]])
    methods = list(metric_values[splitters[0]][datasets[0]])
    representations = list(metric_values[splitters[0]][datasets[0]][methods[0]])
    filename = 'results/figures/benchmark/'+f'BAR_accuracy_{metric}_methods_{x_axis}_{"_".join(datasets)}_{"_".join(methods)}_{"_".join(representations)}'
    if suffix:
        filename += str(suffix)
    font_kwargs = {'family': 'Arial', 'fontsize': 30, "weight": 'bold'}
    if x_axis == "rep":
        n_cols = len(representations)*len(splitters)
    elif x_axis == "algo":
        n_cols = len(methods)*len(splitters)
    else:
        n_cols = len(splitters)
    fig, ax = plt.subplots(1, len(datasets), figsize=(len(datasets)*4, 6), sharey="row")
    axs = np.ravel(ax)
    labels = []
    column_spacing = 4
    for d, dataset_key in enumerate(metric_values[splitters[0]].keys()):
        idx = 0
        for s, splitter in enumerate(metric_values.keys()):
            for i, algo in enumerate(metric_values[splitter][dataset_key].keys()):
                seps = np.linspace(-width*column_spacing*n_cols, width*column_spacing*n_cols, n_cols)
                for j, rep in enumerate(metric_values[splitter][dataset_key][algo].keys()):
                    if x_axis.lower() == "rep":
                        label = rep
                    elif x_axis.lower() == "algo":
                        label = algo
                    else:
                        label = splitter
                    if label not in labels:
                        labels.append(label)
                    mean, std_err = _compute_metric_results(metric_values[splitter][dataset_key][algo][rep][augmentation][metric], metric=metric)
                    mean = mean
                    std_err = std_err
                    selected_color = ac.get(algo) if color_by.lower() == "algo" else rc.get(rep) 
                    if x_axis == "task":
                        selected_color = tc.get(splitter)
                    for k, (_m, _std_err) in enumerate(zip(mean, std_err)):
                        axs[d].bar(seps[idx]+k*0.45, _m, yerr=_std_err, color=selected_color, ecolor="black",
                                    capsize=3, label=label if i == 0 else None,
                                   width=1.2) # NOTE: label reversed mean of quantiles
                    idx += 1

        # Add text elements
        axs[d].text(x=0.25, y=0.05, s="Random CV", horizontalalignment='center', verticalalignment='center',
                 transform=axs[d].transAxes, fontsize=12)
        axs[d].text(x=0.75, y=0.05, s="Position CV", horizontalalignment='center', verticalalignment='center',
                    transform=axs[d].transAxes, fontsize=12)

        # Add vertical/horizontal lines
        axs[d].axvline((seps[int(len(seps)/2)-1]+seps[int(len(seps)/2)])/2, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.5)
        for val in [0, 0.5, 0.75, 0.25, -0.25]:
            axs[d].axhline(val, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.25)

        # x-axis
        if x_axis.lower() == "rep":
            tick_labels = ["One-Hot", "EVE", "EVE (evo-score)", "ProtBert", "ESM"]*2
        elif x_axis.lower() == "algo":
            tick_labels = ["GPlinear", "GPsqexp", "GPm52", "RF", "KNN"]*2
        else:
            tick_labels = labels
        axs[d].set_xticks(seps)
        assert len(seps) == len(tick_labels)
        axs[d].set_xticklabels(tick_labels, size=14, rotation=60, ha='right')
        axs[d].set_xlim((seps[0]-0.75, seps[-1]+0.75+k*0.45))
        axs[d].tick_params(axis='y', which='both', labelsize=22)

        # y-axis
        metric_label = r'$\mathbf{R^2}$' if metric == MSE else SPEARMAN_RHO
        axs[d].set_ylabel(metric_label, **font_kwargs)
        axs[d].yaxis.set_label_coords(-.25, .5, transform=axs[d].transAxes)
        if d > 0:
            axs[d].set_ylabel("")
        axs[d].set_yticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        axs[d].set_ylim((-0.25, 1.0))

        # axs[d].set_title(dataset_key_map[dataset_key], size=25)
    handles, labels = axs[-1].get_legend_handles_labels()
    # fig.legend(handles[:len(labels)], labels[:len(labels)], loc='lower right', ncol=len(labels), prop={'size': 14})
    # plt.suptitle(plot_heading, size=12)
    plt.subplots_adjust(wspace=0.15, left=0.075, right=0.975, bottom=0.25, top=0.95)
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def barplot_metric_comparison_bar_splitting(metric_values: dict, cvtype: str, metric: str, width: float=0.17, color_by: str="algo", x_axis: str="rep", augmentation=None, vline=True, legend=True, n_quantiles=4) -> None:
    filename = 'results/figures/benchmark/'+f'BAR_accuracy_{metric}_methods_{x_axis}'
    font_kwargs = {'family': 'Arial', 'fontsize': 30, "weight": 'bold'}
    font_kwargs_small = {'family': 'Arial', 'fontsize': 18, "weight": 'bold'}
    header_dict = {"1FQG": r"$\beta$-Lactamase", "UBQT": "Ubiquitin"}
    if color_by.lower() not in ["algo", "rep", "task"]:
        warn("Misspecified color-scheme. Defaulting to color by algorithm.")
        color_by = "algo"
    if x_axis.lower() not in ["algo", "rep", "task"]:
        warn("Misspecified the plotting groups. Default to representations.")
        x_axis = "rep"
    splitters = list(metric_values.keys())
    datasets = list(metric_values[splitters[0]])
    methods = list(metric_values[splitters[0]][datasets[0]])
    representations = list(metric_values[splitters[0]][datasets[0]][methods[0]])
    if x_axis == "rep":
        n_cols = len(representations)*len(splitters)
    elif x_axis == "algo":
        n_cols = len(methods)*len(splitters)
    else:
        n_cols = len(splitters)
    fig, ax = plt.subplots(1, len(datasets), figsize=(len(datasets)*4.1, 4))
    axs = np.ravel(ax)
    labels = []
    column_spacing = 4
    for d, dataset_key in enumerate(metric_values[splitters[0]].keys()):
        idx = 0
        for s, splitter in enumerate(metric_values.keys()):
            for i, algo in enumerate(metric_values[splitter][dataset_key].keys()):
                seps = np.linspace(-width*column_spacing*n_cols, width*column_spacing*n_cols, n_cols)
                for j, rep in enumerate(metric_values[splitter][dataset_key][algo].keys()):
                    if x_axis.lower() == "rep":
                        label = rep
                    elif x_axis.lower() == "algo":
                        label = algo
                    else:
                        label = splitter
                    if label not in labels:
                        labels.append(label)
                    mean, std_err = _compute_metric_results(metric_values[splitter][dataset_key][algo][rep][augmentation][metric], metric=metric, n_quantiles=n_quantiles)
                    mean = mean
                    std_err = std_err
                    selected_color = ac.get(algo) if color_by.lower() == "algo" else rc.get(rep) 
                    if x_axis == "task":
                        selected_color = tc.get(splitter)
                    for k, (_m, _std_err) in enumerate(zip(mean, std_err)):
                        axs[d].bar(seps[idx]+k*0.45, _m, yerr=_std_err, color=selected_color, ecolor="black", edgecolor="black", linewidth=2.,
                                    capsize=3, label=label+str(len(mean)-k), alpha=1/mean.shape[-1]) # NOTE: label reversed mean of quantiles
                    idx += 1
        if vline:
            axs[d].axvline((seps[int(len(seps)/2)-1]+seps[int(len(seps)/2)])/2, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.75)
        axs[d].axhline(0, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.5)
        axs[d].axhline(0.5, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.25)
        axs[d].axhline(0.75, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.25)
        axs[d].axhline(0.25, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.25)
        axs[d].axhline(-0.25, seps[0]-0.5, len(splitter)+seps[-1]+0.5, c='grey', ls='--', alpha=0.25)
        axs[d].set_xticks(seps)
        if x_axis.lower() in ["rep", "algo"]:
            tick_labels = [label + "_" + splitter for splitter, label in product(splitters, labels)] 
        elif x_axis.lower() == "task":
            tick_labels = [l.replace("Splitter", "CV").replace("_p15", " ") for l in labels]
        else:
            tick_labels = labels
        assert len(seps) == len(tick_labels)
        axs[d].set_xticklabels(tick_labels, rotation=45, **font_kwargs_small)
        axs[d].xaxis.set_label_coords(-1., .5, transform=axs[d].transAxes)
        axs[d].set_ylim((-0.01, 1.01))
        axs[d].set_xlim((seps[0]-0.75, seps[-1]+0.75+k*0.45))
        axs[d].tick_params(axis='y', which='both', labelsize=22)
        axs[d].set_title(header_dict.get(dataset_key), **font_kwargs)
        if metric == MSE:
            metric_label = r"R$^2$"
        elif metric == SPEARMAN_RHO:
            metric_label = r"spearman $\rho$" 
        else:
            metric_label = metric
        axs[d].set_ylabel(metric_label, **font_kwargs)
    if legend:
        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles[:len(labels)], labels[:len(labels)], loc='lower right', ncol=len(labels), prop={'size': 14})
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.33, left=0.115, right=0.965, bottom=0.27, top=0.9)
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()



def _compute_metric_results(metric_result_list, metric: str, n_quantiles=3):
    obs = np.array(metric_result_list)
    if obs.ndim == 2: # Case: Fraction Splitter or Optimization aggregate results
        if obs.shape[0] < obs.shape[1]: # NOTE: #CV/seed splits << #observations
            obs = np.swapaxes(obs,0,1)
        mu = np.mean(obs, axis=1)
        std_err = np.std(obs, ddof=1, axis=1)/np.sqrt(obs.shape[1]) # std-error on metric across splits/seeds
        if n_quantiles: # in quantiles across all observations
            quantile_len = obs.shape[0]//n_quantiles
            mu = np.mean([mu[i*quantile_len:(i*quantile_len+quantile_len)] for i in range(n_quantiles)], axis=1) # mean across quantiles
            std_err = np.mean([std_err[i*quantile_len:(i*quantile_len+quantile_len)] for i in range(n_quantiles)], axis=1) # std err. across quantiles
        else: # base case, no quantiles
            mu = np.mean(mu)
            std_err = np.mean(std_err)
    else:
        nan_mask = np.isnan(obs)
        if np.sum(nan_mask) > 0:
            print(f"WARNING: {np.sum(nan_mask)} NaNs in metric ({metric}) results")
        mu = np.array([np.mean(obs[~nan_mask])])
        std_err = np.array([np.std(obs[~nan_mask], ddof=1)/np.sqrt(len(obs[~nan_mask]))]) # std-error on metric
    if metric == MSE: # case 1-NMSE, works for both 1D and 2D case
        mu = 1-mu
    return np.array([mu]), np.array([std_err])


def _compute_missing_reference(df, observation_column):
    """
    Utility function to compute missing ranked correlations
    """
    _df = df[[observation_column, "mutation_effect_prediction_vae_ensemble"]].dropna()
    x = _df[observation_column].str.replace(",", ".").astype(float)
    y = _df["mutation_effect_prediction_vae_ensemble"].str.replace(",", ".").astype(float)
    return spearmanr(x,y)[0]


def errorplot_metric_comparison(metric_values: dict, cvtype: str, metric: str, height=0.075, plot_reference=False):
    plot_heading = f'Comparison of algoritms and representations, cv-type: {cvtype} \n scaled, GP optimized zero-mean, var=0.4 (InvGamma(3,3)), len=0.1 (InvGamma(3,3)), noise=0.1 ∈ U[0.01, 1.0]'
    filename = 'results/figures/benchmark/'+'correlation_of_methods_errorbar_'+cvtype+str(list(metric_values.keys()))
    if plot_reference:
        ref_dir = "data/deep_sequence/"
        ref_df = pd.read_excel(f"{ref_dir}41592_2018_138_MOESM6_ESM.xlsx")
        ref_dict = {"1FQG": ref_df[ref_df.dataset=="BLAT_ECOLX_Ranganathan2015"].spearmanr_VAE.values[0],
                    "UBQT": ref_df[ref_df.dataset=="RL401_YEAST_Bolon2013"].spearmanr_VAE.values[0],
                    "MTH3": ref_df[ref_df.protein=="MTH3_HAEAESTABILIZED"].spearmanr_VAE.values[0],
                    "CALM": _compute_missing_reference(pd.read_csv(f"{ref_dir}41592_2018_138_MOESM4_ESM/CALM1_HUMAN_Roth2017.csv", sep=";"), observation_column="screenscore"),
                    "BRCA": _compute_missing_reference(pd.read_csv(f"{ref_dir}41592_2018_138_MOESM4_ESM/BRCA1_HUMAN_BRCT.csv", sep=";"), observation_column="function_score"),
                    "TIMB": _compute_missing_reference(pd.read_csv(f"{ref_dir}41592_2018_138_MOESM4_ESM/TIM_THEMA_b0.csv", sep=";"), observation_column="fitness"),
                    "TOXI": _compute_missing_reference(pd.read_csv(f"{ref_dir}41592_2018_138_MOESM4_ESM/parEparD_Laub2015_all.csv", sep=";"), observation_column="fitness"),
                    }
    fig, ax = plt.subplots(1, len(metric_values.keys()), figsize=(20,6))
    axs = np.ravel(ax)
    reps = []
    for d, dataset_key in enumerate(metric_values.keys()):
        for i, algo in enumerate(metric_values[dataset_key].keys()):
            seps = np.linspace(-height*0.5*len(metric_values[dataset_key].keys()), 
                               height*0.5*len(metric_values[dataset_key].keys()), 
                               len(metric_values[dataset_key][algo].keys()))
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                if rep not in reps:
                    reps.append(rep)
                rho_list = metric_values[dataset_key][algo][rep][None][metric]
                nan_values = np.sum(np.isnan(rho_list))
                rho_list = np.array(rho_list)[np.where(~np.isnan(rho_list))[0]]
                rho_mean = np.mean(rho_list)
                error_on_mean = np.std(rho_list, ddof=1)/np.sqrt(len(rho_list))
                axs[d].errorbar(rho_mean, i+seps[j], xerr=error_on_mean, label=rep, color=rc.get(rep), mec='black', ms=8, capsize=5)
                if bool(nan_values):
                    axs[d].annotate(f"*{nan_values} DNC", xy=(rho_mean+error_on_mean,i+seps[j]))
        if plot_reference and ref_dict.get(dataset_key):
            axs[d].vlines(ref_dict.get(dataset_key), seps[0]-0.25, len(metric_values[dataset_key].keys())-0.25, colors="r", linestyles="dotted", label="DeepSequence")
        axs[d].axvline(0, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
        axs[d].axvline(-1, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
        axs[d].axvline(0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(0.25, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(0.75, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(-0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(-0.25, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(-0.75, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].set_yticks(list(range(len(list(metric_values[dataset_key].keys())))))
        axs[d].set_yticklabels(['' for i in range(len(list(metric_values[dataset_key].keys())))])
        axs[0].set_yticklabels(list(metric_values[dataset_key].keys()), size=16)
        axs[d].set_xlim((-1, 1))
        axs[d].tick_params(axis='x', which='both', labelsize=14)
        axs[d].set_title(dataset_key, size=16)
        axs[d].set_xlabel('spearman r', size=14)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:len(reps)+1], labels[:len(reps)+1], loc='lower right', ncol=len(reps)+1, prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def barplot_metric_augmentation_comparison(metric_values: dict, cvtype: str, augmentation: dict, metric: str, height=0.5, 
                                        dim=None, dim_reduction=LINEAR, reference_values: dict=None):
    plot_heading = f'Augmented models and representations, cv-type: {cvtype.get_name()}, augmentation {str(augmentation)} \n d={dim} {dim_reduction}'
    filename = f'results/figures/augmentation/accuracy_of_methods_barplot_{cvtype.get_name()}_{str(augmentation)}_d={dim}_{dim_reduction}'
    dataset_key = list(metric_values.keys())[0]
    algorithm_key = list(metric_values[dataset_key].keys())[0]
    representation_keys = list(metric_values[dataset_key][algorithm_key].keys())
    fig, axs = plt.subplots(len(metric_values.keys()), len(representation_keys), figsize=(20,15)) # proteins rows, representations columns
    representations = []
    for i, dataset_key in enumerate(metric_values.keys()):
        algorithm_keys = list(metric_values[dataset_key].keys())
        augmentation_keys = list(metric_values[dataset_key][algorithm_key][representation_keys[0]].keys())
        n_bars = len(algorithm_keys)*len(augmentation_keys)
        if reference_values:
            n_bars += len(algorithm_keys) # for each algorithm add one reference value
        seps = np.linspace(-height*0.9*len(algorithm_keys), height*0.9*len(algorithm_keys), n_bars)
        for j, rep in enumerate(representation_keys):
            for k, algo in enumerate(algorithm_keys):
                idx = 0
                for l, aug in enumerate(augmentation_keys):
                    repname = f"{rep}_{aug}"
                    if repname not in representations:
                        representations.append(repname)
                    mse_list = metric_values[dataset_key][algo][rep][aug][metric]
                    neg_invert_mse = 1-np.mean(mse_list)
                    error_on_mean = np.std(mse_list, ddof=1)/np.sqrt(len(mse_list))
                    axs[i,j].errorbar(neg_invert_mse, k+seps[idx], xerr=error_on_mean, color=aug_c.get(aug), ecolor="black",
                                marker=rm.get(rep), fillstyle='none', markersize=14, lw=5, capsize=6, label=repname)
                    axs[i,j].text(neg_invert_mse+0.04, k+seps[idx]+0.03, aug[:1].upper(), fontsize=10) # label with capitalized first letter
                    idx += 1
                if reference_values: # set reference benchmark next to augmented benchmark
                    repname = f"{rep}_reference"
                    representations.append(repname)
                    ref_mse_list = reference_values[dataset_key][algo][rep][None][metric]
                    neg_reference_mse = 1-np.mean(ref_mse_list)
                    ref_error_on_mean = np.std(ref_mse_list, ddof=1)/np.sqrt(len(ref_mse_list))
                    axs[i,j].errorbar(neg_reference_mse, k+seps[idx], xerr=ref_error_on_mean, color="black", ecolor="black",
                            marker=rm.get(rep), fillstyle='none', markersize=14, lw=5, capsize=6, label=repname)
                    idx += 1
                axs[i,j].axhline(k+seps[0]-0.1, -1, 1, c='grey', ls='-', alpha=0.75) # algo separation line
            axs[i,j].axvline(0, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
            axs[i,j].axvline(-1, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
            axs[i,j].axvline(0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
            axs[i,j].axvline(-0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
            axs[i,j].set_yticks(np.arange(seps[0], seps[-1], 1), ['' for i in range(len(list(metric_values[dataset_key].keys())))])
            axs[i,0].set_yticklabels(list(metric_values[dataset_key].keys()), size=16)
            axs[i,j].set_xlim((-1, 1.))
            axs[i,j].tick_params(axis='x', which='both', labelsize=12)
            axs[i,j].set_title(f"{dataset_key} {rep}", size=12)
            axs[i,j].set_xlabel('1 minus normalized MSE', size=12)
    handles, labels = axs[-1,-1].get_legend_handles_labels()
    fig.legend(handles[:len(representations)], labels[:len(representations)], loc='lower right', ncol=len(representations), prop={'size': 9})
    plt.suptitle(plot_heading, size=20)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def threshold_metric_comparison(metric_values: dict, metric: str, datasets: List[str], eps=0.1):
    """
    Figure for threshold comparison across data
    """
    splitter = list(metric_values.keys())[0]
    plot_heading = f"Comparison threshold performance"
    filename = f"results/figures/benchmark/threshold/threshold_comparison_data={datasets[0]}_{metric}"
    for method, data in metric_values.get(splitter).get(datasets[0]).items():
        rep = list(data.keys())[0]
        true_obs_values = data.get(rep).get(None).get(0).get('trues') + data.get(rep).get(None).get(0).get('train_trues') 
        thresholds = np.arange(start=min(true_obs_values)+eps, stop=max(true_obs_values))
        if thresholds[0] < 0 and thresholds[-1] > 0:
            thresholds = np.append(thresholds, 0)
        filtered_results = {}
        for t in thresholds:
            results_dict = filter_functional_variant_data_less_than(metric_values, functional_thresholds=[t])
            filtered_results[t] = results_dict
        break
    n_reps = len(list(metric_values.get(splitter).get(datasets[0]).get('RF').keys())) # TODO: make method name a variable
    fig, ax = plt.subplots(2, n_reps, figsize=(15, 10))
    repnames = []
    for i, (algo, algo_results) in enumerate(metric_values.get(splitter).get(datasets[0]).items()):
        for idx, rep in enumerate(algo_results.keys()):
            ax[1,idx].hist(true_obs_values, int(np.sqrt(len(true_obs_values))), color="black") # histogram of observed values
            if rep not in repnames:
                repnames.append(rep)
            for t_val in thresholds:
                metric_across_splits = []
                for split in filtered_results.get(t_val).get(splitter).get(datasets[0]).get(algo).get(rep).get(None):
                    if MSE == metric[0]:
                        metric_across_splits += filtered_results.get(t_val).get(splitter).get(datasets[0]).get(algo).get(rep).get(None).get(split).get(metric[0].lower())
                    elif SPEARMAN_RHO == metric[0]:
                        metric_across_splits += [spearmanr(filtered_results.get(t_val).get(splitter).get(datasets[0]).get(algo).get(rep).get(None).get(split).get('trues'), 
                        filtered_results.get(t_val).get(splitter).get(datasets[0]).get(algo).get(rep).get(None).get(split).get('pred'))[0]]
                    else:
                        raise ValueError("Misspecified Metric")
                metric_val = 1-np.mean(metric_across_splits) if metric[0] == MSE else np.mean(metric_across_splits) # 1-NMSE or rho
                error_on_mean = np.std(metric_across_splits, ddof=1)/np.sqrt(len(metric_across_splits))
                ax[0,idx].errorbar(t_val+i*0.125, metric_val, yerr=error_on_mean, label=algo, color=ac.get(algo), marker=am.get(algo)) # barplot of performance at threshold
                # TODO: add std-err across splits
                ax[1,idx].set_xlim(min(true_obs_values), max(true_obs_values))
                ax[0,idx].set_xlim(min(true_obs_values), max(true_obs_values))
                ax[0,idx].set_ylim(-1., 1.)
                ax[0,idx].set_xlabel("t threshold value")
                metricname = "1-NMSE" if metric[0] == MSE else "spearman rho"
                ax[0,idx].set_ylabel(metricname)
                ax[0,idx].set_title(rep)
    handles, labels = ax[0,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=len(repnames), prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def pointplot_mutation_comparison(metric_values: dict, x_metric: str, y_metric: str):
    plot_title = f"Mutation Comparison \n {x_metric} against {y_metric} across domains"


def barplot_metric_mutation_comparison(metric_values: dict, metric: str, datasets: List[str], t: float, equality=">", dim=None):
    """
    Figure for mutation plotting for TOXI data
    """
    plot_heading = f'Comparison MUTATION splitting \n d={dim}; t={t} {equality} \n {str(list(metric_values.keys()))}'
    filename = 'results/figures/benchmark/'+f'accuracy_{metric}_of_methods_barplot_d={dim}_t={t}{equality}'+str(list(metric_values.keys()))
    if t: # filter to functional values and keep values <= t
        if equality == "<": # keep values less than threshold
            metric_values = filter_functional_variant_data_less_than(results_dict=metric_values, functional_thresholds=[t])
        elif equality == ">": # keep values larger than threshold
            metric_values = filter_functional_variant_data_greater_than(results_dict=metric_values, functional_thresholds=[t])
        else:
            raise ValueError(f"Specified threshold {t} misspecified equality {equality}!")
    splits = list(metric_values.keys())
    methods = list(metric_values.get(splits[0]).get(datasets[0]).keys())
    representations = list(metric_values[splits[0]][datasets[0]][methods[0]].keys())
    fig, ax = plt.subplots(len(datasets), len(methods), figsize=(len(methods)*6,4), squeeze=False)
    reps = []
    previous_split_keys = []
    n_reps = len(representations)
    width = 0.15+1/(n_reps) # 3 elements (1 bar + 2 arrows) + 2 extra space
    # first set of plots display absolute performance with indicators on previous performance
    all_avrg_metric_vals = []
    all_avrg_metric_errs = []
    for row, dataset in enumerate(datasets):
        for i, algo in enumerate(methods):
            training_variants = []
            testing_variants = []
            for j, splitter_key in enumerate(splits):
                seps = np.linspace(-width*n_reps*len(splits), width*n_reps*len(splits), n_reps*len(splits))
                for k, rep in enumerate(representations):
                    _results_dict = metric_values[splitter_key][dataset][algo][rep][None]
                    if not _results_dict:
                        continue
                    if k==0 and rep != "additive": # collect how many elements in training and test set
                        testing_variants.append(len(_results_dict[list(_results_dict.keys())[0]]['trues']))
                        if 'train_trues' in _results_dict[list(_results_dict.keys())[0]].keys():
                            training_variants.append(len(_results_dict[list(_results_dict.keys())[0]]['train_trues']))
                    k+=j*len(representations)
                    if rep not in reps and "density" not in rep:
                        reps.append(rep)
                    metric_per_split = []
                    for split in _results_dict.keys():
                        if metric == "comparative_NMSE":
                            if j == 0: # compute one baseline for first split, and use this for all MSE computations
                                ref_baseline_mean = np.mean(_results_dict[split].get('train_trues'))
                                ref_baseline_truth = _results_dict[split].get('trues')
                                ref_baseline = np.mean(np.square(ref_baseline_truth - np.repeat(ref_baseline_mean, len(ref_baseline_truth)).reshape(-1,1)))
                            mse_val = mean_squared_error(_results_dict[split]['trues'], _results_dict[split]['pred'])
                            metric_per_split.append(1-mse_val/ref_baseline)
                        elif metric == MSE: # Make 1-NMSE => R2
                            mean_y = np.mean(_results_dict[split].get('train_trues'))
                            baseline = np.mean(np.square(_results_dict[split]['trues'] - np.repeat(mean_y, len(_results_dict[split]['trues'])).reshape(-1,1)))
                            if 'mse' in _results_dict[split].keys():
                                metric_per_split.append(1-np.mean(_results_dict[split][MSE])) # NOTE: values recorded as MSE have been normalized >> np.mean(err2)/baseline <<
                            elif 'mse' not in _results_dict[split].keys():
                                mse_val = mean_squared_error(_results_dict[split]['trues'], _results_dict[split]['pred'])
                                # TODO: determine why additive benchmark has different test set!!
                                metric_per_split.append(1-mse_val/baseline)
                            else:
                                raise ValueError(f"Experiment MSE not computed for {dataset}, {algo}, {rep}, {splitter_key}")
                        elif metric == SPEARMAN_RHO:
                            trues = np.array(_results_dict[split]['trues'])
                            pred = np.array(_results_dict[split]['pred'])
                            metric_per_split.append(spearmanr(trues, pred)[0])
                        else: # BASECASE: regular MSE
                            trues = np.array(_results_dict[split]['trues'])
                            pred = np.array(_results_dict[split]['pred'])
                            metric_per_split.append(mean_squared_error(trues, pred))
                    _metric_val = np.mean(metric_per_split)
                    _metric_std_err = np.std(metric_per_split, ddof=1)/np.sqrt(len(metric_per_split)) if len(metric_per_split) > 1 else 0.
                    all_avrg_metric_vals.append(_metric_val)
                    all_avrg_metric_errs.append(_metric_std_err)
                    ax[row, i].bar(j+seps[k], _metric_val, yerr=_metric_std_err, width=width, label=rep, color=rc.get(rep),
                                    facecolor=rc.get(rep), edgecolor="k", ecolor='black', capsize=5)
            previous_split_keys.append(splitter_key)
            cols = len(splits)
            abs_min, abs_max = min(all_avrg_metric_vals)-max(all_avrg_metric_errs), max(all_avrg_metric_vals)+max(all_avrg_metric_errs)
            abs_min = abs_min if abs_min < 0. else 0.
            # main markers:
            ax[row, i].axhline(0., seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.75)
            for x in np.arange(-30, 15.1, 1):
                ax[row, i].axhline(x, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.5)
            # secondary markers:
            for x in np.arange(-30, 15.1, 0.5):
                ax[row, i].axhline(x, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.125)
            for x in np.arange(-30, 15.1, 0.25):
                ax[row, i].axhline(x, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.025)
            # _ax.set_xticks([x for x in range(len(splits))])
            # _ax.set_xticklabels([f"{split} \n frac.: {n}/{c}" for split, n, c in zip(splits, training_variants, N_combinations)])
            if metric == SPEARMAN_RHO:
                ax[row, i].set_ylim((-0.251, 1.1))
            elif metric == MSE or metric == "comparative_NMSE":
                ax[row, i].set_ylim((-3, 1.1))
            else:
                ax[row, i].set_ylim((-0.1, 3))
            ax[row, i].tick_params(axis='x', which='both', labelsize=9)
            metric_name = "R2" if metric == MSE else metric
            metric_name = metric_name if metric else "MSE" # base-case
            ax[row, i].set_ylabel(metric_name, fontsize=21)
            ax[row, i].set_title(f"{algo} - {dataset}\n{testing_variants} \n {metric}")
    handles, labels = ax[row, i].get_legend_handles_labels()
    fig.legend(handles[:len(labels)], reps[:len(labels)], loc='lower right', ncol=len(reps), prop={'size': 14}) # handles[:len(reps)]
    plt.suptitle(plot_heading, size=12)
    #plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def barplot_metric_mutation_matrix(metric_values: dict, metric: str, datasets: List[str], dim=None):
    """
    Figure for mutation plotting for TOXI data
    """
    title_dict = {"BioSplitter1_1": "1M", 
                "BioSplitter2_2": r"$\leq$ 2M", 
                "BioSplitter3_3": r"$\leq$ 3M", 
                "BioSplitter1_2": "2M", 
                "BioSplitter2_3": "3M",
                "BioSplitter3_4": "4M"}
    font_kwargs = {'family': 'Arial', 'fontsize': 31, "weight": 'bold'}
    plot_dists = False if 'base_mse' in metric.lower() else True # NOTE: if accuracy, do not plot left hand densities
    filename = 'results/figures/benchmark/'+f'matrix_{metric}_of_mutations_barplot_d={dim}'+str(list(metric_values.keys()))
    LW = 5.
    splits = list(metric_values.keys())
    dataset = list(metric_values.get(splits[0]).keys())[0]
    method = list(metric_values.get(splits[0]).get(datasets[0]).keys())[0]
    representations = list(metric_values[splits[0]][dataset][method].keys())
    fig, ax = plt.subplots(int(len(splits)/2+1), int(len(splits)/2+2), figsize=(11.5, 9), squeeze=False)
    reps = []
    n_reps = len(representations)
    width = 3+1/(n_reps) # 3 elements (1 bar + 2 arrows) + 2 extra space
    ax[0, 0].axis('off')
    for row, col in product(range(1,4), range(1,5)):
        splitter_key = f"BioSplitter{row}_{col}"
        if row>0 and col>0 and row!=col and row+1!=col:
            ax[row, col].axis('off')
            continue
        if splitter_key not in splits:
            continue
        if row == col: # joint distribution is all available data for 1->2;2->3;3->4
            joint_obs_vals = metric_values[splitter_key][dataset][method][representations[0]][None][0].get('train_trues') +  metric_values[splitter_key][dataset][method][representations[0]][None][0].get('trues')
            sns.kdeplot(-np.array(joint_obs_vals), ax=ax[row, 0], color=rc.get(representations[0]), linewidth=LW)
            if row == 1:
                sns.kdeplot(-np.array(joint_obs_vals), ax=ax[0, col], color=rc.get(representations[0]), linewidth=LW) # 1M is symmetric
                ax[0, col].set_title(title_dict.get(splitter_key), **font_kwargs)
                ax[0, col].set_ylabel("")
                ax[0, col].tick_params(axis='y', which='both', labelsize=20)
                ax[0, col].tick_params(axis='x', which='both', labelsize=20)
            ax[row, 0].set_xlim((-0.5, 1.5))
            #ax[row, 0].set_title(title_dict.get(splitter_key), **font_kwargs)
            ax[row, 0].set_ylabel(title_dict.get(splitter_key), **font_kwargs)
            ax[row, 0].tick_params(axis='y', which='both', labelsize=20)
            ax[row, 0].tick_params(axis='x', which='both', labelsize=20)
        if row == col and not plot_dists:
            fig.delaxes(ax[row, 0])
        if row+1 == col:
            obs_vals = metric_values[splitter_key][dataset][method][representations[0]][None][0].get('trues') # Test distribution on
            sns.kdeplot(-np.array(obs_vals), ax=ax[0,col], color=rc.get(representations[0]), linewidth=LW)
            ax[0, col].set_xlim((-0.5, 1.5))
            ax[0, col].set_title(title_dict.get(splitter_key), **font_kwargs)
            ax[0, col].set_ylabel("")
            ax[0, col].tick_params(axis='y', which='both', labelsize=20)
            ax[0, col].tick_params(axis='x', which='both', labelsize=20)
        seps = np.linspace((-width*n_reps)/3, (width*n_reps)/3, n_reps-1)
        for k, rep in enumerate(representations):
            _results_dict = metric_values[splitter_key][dataset][method][rep][None]
            metric_per_split = []
            for s in _results_dict.keys():
                if rep not in reps:
                    reps.append(rep)
                if metric == MSE: # Make 1-NMSE => R2
                    mean_y = np.mean(_results_dict[s].get('train_trues'))
                    baseline = np.mean(np.square(_results_dict[s]['trues'] - np.repeat(mean_y, len(_results_dict[s]['trues'])).reshape(-1,1)))
                    if 'mse' in _results_dict[s].keys():
                        metric_per_split.append(1-np.mean(_results_dict[s][MSE])) # NOTE: values recorded as MSE have been normalized >> np.mean(err2)/baseline <<
                    elif 'mse' not in _results_dict[s].keys():
                        mse_val = mean_squared_error(_results_dict[s]['trues'], _results_dict[s]['pred'])
                        metric_per_split.append(1-mse_val/baseline)
                    else:
                        raise ValueError(f"Experiment MSE not computed for {dataset}, {method}, {rep}, {splits[k]}")
                elif metric == SPEARMAN_RHO:
                    trues = np.array(_results_dict[s]['trues'])
                    pred = np.array(_results_dict[s]['pred'])
                    metric_per_split.append(spearmanr(trues, pred)[0])
                else: # BASECASE: regular MSE
                    trues = np.array(_results_dict[s]['trues'])
                    pred = np.array(_results_dict[s]['pred'])
                    metric_per_split.append(mean_squared_error(trues, pred))
            _metric_val = np.mean(metric_per_split)
            _metric_std_err = np.std(metric_per_split, ddof=1)/np.sqrt(len(metric_per_split)) if len(metric_per_split) > 1 else 0.
            if rep == "additive" and bool(_metric_val): #NOTE: 1M rank-corr is NaN for constant additive values.
                if metric == SPEARMAN_RHO:
                    ax[row, col].plot(seps, np.repeat(_metric_val, len(seps)), color=rc.get(rep), label=rep, linestyle="dashed", lw=3.)
                # else:
                #     ax[row, col].text(seps[0], -0.05, f"add.={np.round(_metric_val,2)}", **font_kwargs_small)
            else:
                ax[row, col].bar(seps[k], _metric_val, yerr=_metric_std_err, width=width, label=rep, color=rc.get(rep),
                            facecolor=rc.get(rep), edgecolor="k", ecolor='black', capsize=7)
            ax[row, col].tick_params(axis='y', which='both', labelsize=20)
            ax[row, col].tick_params(axis='x', which='both', labelbottom=False, bottom=False)
            if metric == SPEARMAN_RHO:
                ax[row, col].set_ylim(-0.2, 1.01)
            if "MSE" in metric and metric != MSE:
                ax[row, col].set_ylim(-0.01, 0.4)
            if row==col:
                if metric == MSE:
                    metric_name = r"$R^2$"
                elif metric == "base_MSE":
                    metric_name = "mse" 
                else:
                    metric_name = metric
                # metric_name = metric_name if metric else "MSE" # base-case
                if metric == SPEARMAN_RHO:
                    ax[row, col].set_ylabel(r"spearman $\rho$", **font_kwargs)
                else:
                    ax[row, col].set_ylabel(metric_name, **font_kwargs)
    handles, labels = ax[row, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(reps), prop={'size': 19})
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.38, left=0.055, right=0.99, bottom=0.11, top=0.95)
    if not plot_dists:
        plt.subplots_adjust(wspace=0.38, left=0., right=0.99, bottom=0.11, top=0.95)
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def scatterplot_metric_threshold_comparison(metric_values: dict, metric: str=None, dim=None, datasets: List[str]=["TOXI"], thresholds=None, N_combinations=[]):
    """
    Prediction plotting given thresholds
    """
    plot_heading = f'Predictions given t={thresholds}'
    filename = f'results/figures/benchmark/threshold_{metric}_of_methods_barplot_d={dim}_t={thresholds}_{datasets}'+str(list(metric_values.keys()))
    splits = list(metric_values.keys())
    methods = list(metric_values.get(splits[0]).get(datasets[0]).keys())
    representations = list(metric_values[splits[0]][datasets[0]][methods[0]].keys())
    fig, ax = plt.subplots(len(datasets), len(methods), figsize=(len(methods)*6,6.5))
    reps = []
    previous_split_keys = []
    n_reps = len(representations)
    width = 0.15+1/(n_reps) # 3 elements (1 bar + 2 arrows) + 2 extra space
    # first set of plots display absolute performance with indicators on previous performance
    for row, dataset in enumerate(datasets):
        for i, algo in enumerate(methods):
            plt_idx = (row, i) if len(datasets) > 1 else i
            training_variants = []
            testing_variants = []
            mse_vals = []
            for j, splitter_key in enumerate(splits):
                spearman_rs = []
                for k, rep in enumerate(representations):
                    _results_dict = metric_values[splitter_key][dataset][algo][rep][None]
                    if k==0 and rep != "additive": # collect how many elements in training and test set
                        testing_variants.append(len(_results_dict[0]['trues']))
                        if 'train_trues' in _results_dict[0].keys():
                            training_variants.append(len(_results_dict[0]['train_trues']))
                    k+=j*len(representations)
                    if rep not in reps and "density" not in rep:
                        reps.append(rep)
                    pred_per_rep = []
                    true_per_rep = []
                    for split in _results_dict.keys():
                        true_per_rep += _results_dict[split]['trues']
                        pred_per_rep += _results_dict[split]['pred']
                        if MSE.lower() in _results_dict[split].keys():
                            mse_vals += _results_dict[split]['mse']
                    ax[plt_idx].scatter(true_per_rep, pred_per_rep, 
                                label=rep, color=rc.get(rep))
                    r, _ = spearmanr(true_per_rep, pred_per_rep)
                    spearman_rs.append(np.round(r, 2))
            previous_split_keys.append(splitter_key)
            cols = len(splits)
            # ax[plt_idx].set_xticks([x for x in range(len(splits))])
            # ax[plt_idx].set_xticklabels([f"{split} \n frac.: {n}/{c}" for split, n, c in zip(splits, training_variants, N_combinations)])
            ax[plt_idx].tick_params(axis='x', which='both', labelsize=9)
            # ax[plt_idx].set_xlabel(algo, size=14)
            ax[plt_idx].set_ylabel("prediction")
            ax[plt_idx].set_xlabel("observation")
            rep_to_r = [(rep, r) for rep, r in zip(representations, spearman_rs)]
            ax[plt_idx].set_title(f"{algo} - {dataset}\n{rep_to_r}\nMSE:{np.round(np.mean(mse_vals), 2)}\n{testing_variants}")
    handles, labels = ax[plt_idx].get_legend_handles_labels()
    fig.legend(handles[:len(reps)], labels[:len(reps)], loc='lower right', ncol=len(reps), prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def barplot_metric_functional_mutation_comparison(metric_values: dict, metric: str='mse', dim=None):
    # TODO: code duplication with function above CLEAN-UP!
    plot_heading = f'Comparison of algoritms for FUNCTIONAL MUTATION splitting \n t=-0.5, d={dim} scaled, GP optimized zero-mean, var=0.4 (InvGamma(3,3)), len=0.1 (InvGamma(3,3)), noise=0.1 ∈ [0.01, 1.0] (Uniform)'
    filename = 'results/figures/benchmark/functional/'+f'functional_acc_of_methods_barplot_d={dim}_'+str(list(metric_values.keys()))
    fig, ax = plt.subplots(1, len(metric_values.keys()), figsize=(len(metric_values.keys())*4,6))
    axs = np.ravel(ax)
    reps = []
    previous_split_keys = []
    n_reps = 5 # OR: len of second level metrics dictionary
    height = 1/(n_reps*5) # 3 elements (1 bar + 2 arrows) + 2 extra space
    # first set of plots display absolute performance with indicators on previous performance
    for d, splitter_key in enumerate(metric_values.keys()):
        for dataset_key in metric_values[splitter_key].keys():
            for i, algo in enumerate(metric_values[splitter_key][dataset_key].keys()):
                seps = np.linspace(-height*n_reps, height*n_reps, n_reps)
                for j, rep in enumerate(metric_values[splitter_key][dataset_key][algo].keys()):
                    if rep not in reps and "density" not in rep:
                        reps.append(rep)
                    mse_list = metric_values[splitter_key][dataset_key][algo][rep][0][metric]
                    neg_invert_mse = 1-np.mean(mse_list)
                    previous_metric = 1-np.mean(metric_values[previous_split_keys[-1]][dataset_key][algo][rep][0][metric]) if len(previous_split_keys) > 0 else 0.
                    prev_previous_metric = 1-np.mean(metric_values[previous_split_keys[-2]][dataset_key][algo][rep][0][metric]) if len(previous_split_keys) > 1 else 0.
                    if rep in [VAE_DENSITY, EVE_DENSITY]: # overlay VAE density as reference on VAE row
                        if rep == VAE_DENSITY:
                            ref = VAE
                        elif rep == EVE_DENSITY:
                            ref = EVE
                        pos = list(metric_values[splitter_key][dataset_key][algo].keys()).index(ref)
                        axs[d].boxplot(np.ones(len(mse_list)) - mse_list, positions=[i+seps[pos]], widths=[height], labels=[rep], vert=False)
                    else: # if improvement, plot previous shaded and improvement solid
                        if neg_invert_mse > previous_metric:
                            axs[d].barh(i+seps[j], neg_invert_mse - previous_metric, left=previous_metric, height=height, label=rep, color=rc.get(rep),
                                        facecolor=rc.get(rep), edgecolor=rc.get(rep), ecolor='black', capsize=5, hatch='//')
                            axs[d].barh(i+seps[j], neg_invert_mse, height=height, color=rc.get(rep), alpha=0.125,
                                        facecolor=rc.get(rep), edgecolor=rc.get(rep), ecolor='black', capsize=5, hatch='//')
                        else: # if worse: plot diff to previous performance shaded and current performance solid
                            axs[d].barh(i+seps[j], neg_invert_mse - previous_metric, left=previous_metric, height=height, color=rc.get(rep),
                                        facecolor=rc.get(rep), edgecolor="red", ecolor='black', capsize=5, hatch='//', alpha=0.125)
                            axs[d].barh(i+seps[j], neg_invert_mse, height=height, label=rep, color=rc.get(rep),
                                        facecolor=rc.get(rep), edgecolor=rc.get(rep), ecolor='black', capsize=5, hatch='//')
                        # mark diff explicitly with arrow:
                        if d > 0: # mark difference to previous explicitly as error
                            performance_diff_to_prev = previous_metric+(neg_invert_mse-previous_metric)
                            if performance_diff_to_prev < -0.99: # cap arrows to xlim
                                axs[d].annotate("", xy=(previous_metric, i+seps[j]+height*0.1), 
                                                xytext=(-1.1, i+seps[j]+height*0.1), 
                                                arrowprops=dict(arrowstyle="-"))
                            else:
                                axs[d].annotate("", xy=(previous_metric, i+seps[j]+height*0.1), 
                                                xytext=(performance_diff_to_prev, i+seps[j]+height*0.1), 
                                                arrowprops=dict(arrowstyle="<-"))
                        if d > 1:
                            performance_diff = prev_previous_metric+(neg_invert_mse-prev_previous_metric)
                            if performance_diff < -0.99: # cap arrows to xlim
                                axs[d].annotate("", xy=(prev_previous_metric, i+seps[j]+height*1.5), 
                                                xytext=(-1.1, i+seps[j]+height*1.5), 
                                                arrowprops=dict(arrowstyle="-", linestyle="-", color=rc.get(rep)))
                            else:
                                axs[d].annotate("", xy=(prev_previous_metric, i+seps[j]+height*1.5), 
                                                xytext=(performance_diff, i+seps[j]+height*1.5), 
                                                arrowprops=dict(arrowstyle="<-", linestyle="-", color=rc.get(rep)))
        previous_split_keys.append(splitter_key)
        cols = len(metric_values[splitter_key][dataset_key].keys())
        axs[d].axvline(0, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.5)
        axs[d].axvline(-1, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.5)
        axs[d].axvline(1, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.5)
        axs[d].axvline(0.75, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].axvline(0.5, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(0.25, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].axvline(-0.5, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.25)
        axs[d].axvline(-0.25, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].axvline(-0.75, seps[0], cols-1+seps[-1], c='grey', ls='--', alpha=0.125)
        axs[d].set_yticks(list(range(len(list(metric_values[splitter_key][dataset_key].keys())))))
        axs[d].set_yticklabels(['' for i in range(len(list(metric_values[splitter_key][dataset_key].keys())))])
        axs[0].set_yticklabels(list(metric_values[splitter_key][dataset_key].keys()), size=16)
        axs[d].set_xlim((-1.1, 1.1))
        axs[d].tick_params(axis='x', which='both', labelsize=14)
        axs[d].set_title(splitter_key, size=16)
        axs[d].set_xlabel('1-NMSE', size=14)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:len(reps)], reps, loc='lower right', ncol=len(reps), prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def plot_optimization_task(metric_values: dict, representation: str, dataset: list, name: str, max_iterations=500, legend=False):
    plt.figure()
    font_kwargs = {'family': 'Arial', 'fontsize': 30, "weight": 'bold'}
    header_dict = {"1FQG": r"$\beta$-Lactamase", "UBQT": "Ubiquitin"}
    for d, dataset_key in enumerate(metric_values.keys()):
        algos = []
        for i, algo in enumerate(metric_values[dataset_key].keys()):           
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                if algo not in algos:
                    algos.append(algo)
                observations = np.vstack(metric_values[dataset_key][algo][rep][-max_iterations:])
                means = np.mean(observations, axis=0)
                stds = np.std(observations, ddof=1, axis=0)/np.sqrt(observations.shape[0])
                plt.plot(means, color=ac.get(algo), label=algo, linewidth=4)
                plt.fill_between(list(range(len(means))), means-stds, means+stds, color=ac.get(algo), alpha=0.5)
                if 'best' in name.lower():
                    _, Y = load_dataset(dataset_key, representation=ONE_HOT)
                    plt.hlines(min(Y), 0, len(means), linestyles='--', linewidth=2.5, colors='dimgrey')
    plt.xlabel('Iterations', **font_kwargs)
    plt.ylabel('observed value', **font_kwargs)
    if 'best' in name.lower():
        if 'fqg' in dataset[0].lower():
            plt.ylim(-0.4, 0.2) # TODO: set this value dynamically
        else:
            plt.ylim(-0.1, 0.2)
    if 'regret' in name.lower():
        plt.ylabel('cumulative regret', **font_kwargs)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.title(f"{representation} {header_dict.get(dataset[0].upper())}", **font_kwargs)
    markers = [plt.Line2D([0,0],[0,0],color=ac.get(algo), marker='o', linestyle='') for algo in algos]
    if legend:
        plt.legend(markers, algos, loc="lower right", numpoints=1, ncol=len(algos), prop={'size':14})
    plt.tight_layout()
    plt.savefig('results/figures/optim/'+name+dataset[0]+representation+'_optimization_plot.png', bbox_inches="tight")
    plt.savefig('results/figures/optim/'+name+dataset[0]+representation+'_optimization_plot.pdf', bbox_inches="tight")
    plt.show()


def __parse_cumulative_results_dict(metrics_values: dict, metric: str, number_quantiles=10) -> dict:
    data_fractions = list(metrics_values.keys())
    dataset = list(metrics_values[data_fractions[0]].keys())[0]
    methods = list(metrics_values[data_fractions[0]][dataset].keys())
    representations = list(metrics_values[data_fractions[0]][dataset][methods[0]].keys())
    observations = {m: {'mean': [], 'std': [], 'std_err': [], 'eps': [],
                        'mean_ece': [], 'ece_err': [], 'sharpness': []} for m in methods}
    for fraction in data_fractions:
        for method in methods:
            for representation in representations:
                _results = metrics_values[fraction][dataset][method][representation][None]
                number_splits = len(list(_results.keys()))
                _metric = [_results[s][metric] for s in range(number_splits)]
                _eps = [_results[s].get(PAC_BAYES_EPS) for s in range(number_splits)]
                if metric == MSE: # R2 metric
                    mean_metric = 1-np.mean(_metric)
                elif metric == SPEARMAN_RHO:
                    mean_metric = np.mean(_metric)
                else:
                    mean_metric = np.mean(_metric)
                std_metric = np.std(_metric)
                observations[method]['mean'].append(mean_metric)
                observations[method]['std'].append(std_metric)
                observations[method]['std_err'].append(std_metric/np.sqrt(len(_metric)))
                if all(_eps) and (metric is MSE or metric is SPEARMAN_RHO):
                    epsilon_observation = np.mean(_eps)
                    observations[method]['eps'].append(epsilon_observation)
                ece = []
                sharpness = []
                for s in _results.keys():
                    trues = _results[s]['trues']
                    preds = _results[s]['pred']
                    uncertainties = _results[s]['unc']
                    data_uncertainties = _results[s].get(GP_L_VAR)
                    if data_uncertainties:
                        _scale_std = _results[s].get(STD_Y)
                        uncertainties += np.sqrt(data_uncertainties*_scale_std)
                    _, _, e, s = prep_reliability_diagram(trues, preds, uncertainties, number_quantiles)
                    ece.append(e)
                    sharpness.append(s)
                observations[method]['mean_ece'].append(np.mean(ece))
                observations[method]['ece_err'].append(np.std(ece) / np.sqrt(len(_metric))) # standard error
                observations[method]['sharpness'].append(np.mean(sharpness))
    return observations


def cumulative_performance_plot(metrics_values: dict, metrics=[MLL, MSE, SPEARMAN_RHO], number_quantiles=10, threshold=None):
    header_dict = {"1FQG": r"$\beta$-Lactamase", "UBQT": "Ubiquitin", "CALM": "Calmodulin", 
                    "TIMB": "TIM-Barrel", "MTH3": "MTH3", "BRCA": "BRCA"}
    header_rep_dict = {"one_hot": "One-Hot", "transformer": "ProtBert", "eve": "EVE", "esm": "ESM"}
    font_kwargs = {'family': 'Arial', 'fontsize': 28, "weight": 'bold'}
    font_kwargs_small = {'family': 'Arial', 'fontsize': 18, "weight": 'bold'}
    data_fractions = np.array(list(metrics_values.keys())).astype(float)
    dataset = list(metrics_values[data_fractions[0]].keys())[0]
    methods = list(metrics_values[data_fractions[0]][dataset].keys())
    representation = list(metrics_values[data_fractions[0]][dataset][methods[0]].keys())[0]
    for metric in metrics:
        observations = __parse_cumulative_results_dict(metrics_values=metrics_values, metric=metric, number_quantiles=number_quantiles)
        fig, ax = plt.subplots(2, figsize=(7,5))
        for method in methods:
            # TODO: keep index of NaNs and annotate with stars?
            # y = np.nan_to_num(np.array(observations[method]['mean']))
            # ece = np.nan_to_num(np.array(observations[method]['mean_ece']))
            yerr = np.array(observations[method]['std_err'])
            ece_err = np.array(observations[method]['ece_err'])
            y = np.array(observations[method]['mean'])
            ece = np.array(observations[method]['mean_ece'])
            if PAC_BAYES_EPS in metrics:
                bound = np.nan_to_num(np.array(observations[method]['eps']))
            else:
                #bound = np.nan_to_num(np.array(observations[method]['std']))
                bound = np.array(observations[method]['std'])
            ax[0].errorbar(data_fractions, y, yerr=yerr, lw=3, color=ac.get(method), label=method)
            if metric == MLL:
                ax[0].set_yscale('log')
            if metric in [MSE, SPEARMAN_RHO] and method not in ['RF', 'KNN']:
                label = r"PAC $\lambda$-bound $\delta$=.05" if PAC_BAYES_EPS in metrics else "std"
                ax[0].fill_between(data_fractions, y+bound, y-bound, alpha=0.2, color=ac.get(method), label=label)
                ax[0].set_ylim((-0.3, 1))
            ax[1].errorbar(data_fractions, ece, yerr=ece_err, lw=3, color=ac.get(method), label=method)
        ax[1].set_xlabel('fraction of N', **font_kwargs)
        if metric == MSE:
            metric = r"$R^2$"
        elif metric == SPEARMAN_RHO:
            metric = r"$\rho$"
        ax[0].set_ylabel(f'{metric}', **font_kwargs)
        ax[1].set_ylabel('ECE', **font_kwargs)
        title_string = f"{header_dict.get(dataset)} ({header_rep_dict.get(representation)})"
        # if threshold:
        #     title_string += f" t={threshold}"
        plt.suptitle(title_string, **font_kwargs)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/figures/fraction_benchmark/{dataset}_{metric.replace("$", "").replace("^", "")}_{representation}.png')
        plt.savefig(f'results/figures/fraction_benchmark/{dataset}_{metric.replace("$", "").replace("^", "")}_{representation}.pdf')
        plt.show()
