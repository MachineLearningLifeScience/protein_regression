from distutils.log import error
import enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from data.load_dataset import load_dataset
from visualization.plot_metric_for_uncertainties import prep_reliability_diagram
from visualization import algorithm_colors as ac
from visualization import colorscheme2 as cc
from visualization import representation_colors as rc
from util.mlflow.constants import GP_L_VAR, LINEAR, VAE, EVE, VAE_DENSITY, ONE_HOT, EVE_DENSITY
from util.mlflow.constants import MLL, MSE, SPEARMAN_RHO, PAC_BAYES_EPS, STD_Y


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
                if rep not in reps and "density" not in rep:
                    reps.append(rep)
                mse_list = metric_values[dataset_key][algo][rep][None][metric]
                neg_invert_mse = 1-np.mean(mse_list)
                error_on_mean = np.std(mse_list, ddof=1)/np.sqrt(len(mse_list))
                if rep in [VAE_DENSITY, EVE_DENSITY]: # overlay VAE density as reference on VAE row
                    if rep == VAE_DENSITY:
                        ref = VAE
                    elif rep == EVE_DENSITY:
                        ref = EVE
                    pos = list(metric_values[dataset_key][algo].keys()).index(ref)
                    axs[d].boxplot(np.ones(len(mse_list)) - mse_list, positions=[i+seps[pos]], widths=[height], labels=[rep], vert=False)
                else:
                    axs[d].barh(i+seps[j], neg_invert_mse, xerr=error_on_mean, height=height, label=rep, color=rc.get(rep), 
                                facecolor=rc.get(rep), edgecolor=rc.get(rep), ecolor='black', capsize=5, hatch='//')
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
        axs[0].set_yticklabels(list(metric_values[dataset_key].keys()), size=16)
        axs[d].set_xlim((-1, 1))
        axs[d].tick_params(axis='x', which='both', labelsize=14)
        axs[d].set_title(dataset_key, size=16)
        axs[d].set_xlabel('1-NMSE', size=14)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:len(reps)], reps, loc='lower right', prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def errorplot_metric_comparison(metric_values: dict, cvtype: str, metric: str, height=0.075, plot_reference=False):
    plot_heading = f'Comparison of algoritms and representations, cv-type: {cvtype} \n scaled, GP optimized zero-mean, var=0.4 (InvGamma(3,3)), len=0.1 (InvGamma(3,3)), noise=0.1 ∈ U[0.01, 1.0]'
    filename = 'results/figures/benchmark/'+'correlation_of_methods_errorbar_'+cvtype+str(list(metric_values.keys()))
    if plot_reference:
        ref_df = pd.read_excel("data/riesselman_ref/41592_2018_138_MOESM6_ESM.xlsx")
        ref_dict = {"1FQG": ref_df[ref_df.dataset=="BLAT_ECOLX_Ranganathan2015"].spearmanr_VAE.values[0],
                    "UBQT": ref_df[ref_df.dataset=="RL401_YEAST_Bolon2013"].spearmanr_VAE.values[0],
                    "MTH3": ref_df[ref_df.protein=="MTH3_HAEAESTABILIZED"].spearmanr_VAE.values[0],
                    #"BRCA": ref_df[ref_df.dataset=="BRCA1_HUMAN_Fields2015" & ref_df.feature=="hdr"].spearmanr_VAE.values[0],
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
                rho_mean = np.mean(rho_list)
                error_on_mean = np.std(rho_list, ddof=1)/np.sqrt(len(rho_list))
                axs[d].errorbar(rho_mean, i+seps[j], xerr=error_on_mean, label=rep, color=rc.get(rep), mec='black', ms=8, capsize=5)
        if plot_reference and ref_dict.get(dataset_key):
            axs[d].vlines(ref_dict.get(dataset_key), seps[0], len(metric_values[dataset_key].keys())+seps[-1], colors="k", linestyles="dotted", label="DeepSequence")
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
    fig.legend(handles[:len(reps)], reps, loc='lower right', prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def barplot_metric_augmentation_comparison(metric_values: dict, cvtype: str, augmentation: dict, metric: str, height=0.3, 
                                        dim=None, dim_reduction=LINEAR, reference_values: dict=None):
    plot_heading = f'Augmented models and representations, cv-type: {cvtype}, augmentation {str(augmentation)} \n d={dim} {dim_reduction}'
    filename = f'results/figures/augmentation/accuracy_of_methods_barplot_{cvtype}_{str(augmentation)}_d={dim}_{dim_reduction}'
    fig, ax = plt.subplots(1, len(metric_values.keys()), figsize=(20,5))
    axs = np.ravel(ax)
    representations = []
    for i, dataset_key in enumerate(metric_values.keys()):
        algorithm_keys = metric_values[dataset_key].keys()
        for j, algo in enumerate(algorithm_keys):
            representation_keys = metric_values[dataset_key][algo].keys()
            idx = 0
            for k, rep in enumerate(representation_keys):
                augmentation_keys = metric_values[dataset_key][algo][rep].keys()
                seps = np.linspace(-height*0.7*len(algorithm_keys), height*0.7*len(algorithm_keys), 
                                len(algorithm_keys)*len(representation_keys)*len(augmentation_keys))
                for l, aug in enumerate(augmentation_keys):
                    repname = f"{rep}_{aug}"
                    if repname not in representations:
                        representations.append(repname)
                    mse_list = metric_values[dataset_key][algo][rep][aug][metric]
                    neg_invert_mse = 1-np.mean(mse_list)
                    error_on_mean = np.std(mse_list, ddof=1)/np.sqrt(len(mse_list))
                    if reference_values: # overlay by mean reference benchmark
                        neg_reference_mse = 1-np.mean(reference_values[dataset_key][algo][rep][None][metric])
                        axs[i].barh(j+seps[idx], neg_reference_mse-neg_invert_mse, label=repname, xerr=error_on_mean, height=height*0.25, color=rc.get(rep), alpha=0.8,
                                    facecolor=rc.get(rep), edgecolor=cc[l], ecolor='black', capsize=5, hatch='/', linewidth=2)
                    else:
                        axs[i].barh(j+seps[idx], neg_invert_mse, xerr=error_on_mean, height=height*0.25, label=repname, color=rc.get(rep), 
                            facecolor=rc.get(rep), edgecolor=cc[l], ecolor='black', capsize=5, hatch='/', linewidth=2)
                    axs[i].axvline(0, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
                    axs[i].axvline(-1, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
                    axs[i].axvline(0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
                    axs[i].axvline(-0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
                    axs[i].set_yticks(np.arange(0, len(list(metric_values[dataset_key].keys())))-0.5, ['' for i in range(len(list(metric_values[dataset_key].keys())))])
                    axs[0].set_yticklabels(list(metric_values[dataset_key].keys()), size=16)
                    axs[i].set_xlim((-1, 1.))
                    axs[i].tick_params(axis='x', which='both', labelsize=12)
                    axs[i].set_title(dataset_key, size=12)
                    axs[i].set_xlabel('1 minus normalized MSE', size=12)
                    idx += 1
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:len(representations)], representations, loc='lower right', ncol=2, prop={'size': 10})
    plt.suptitle(plot_heading, size=20)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def barplot_metric_mutation_comparison(metric_values: dict, metric: str=MSE):
    plot_heading = f'Comparison of algoritms and representations for MUTATION splitting \n scaled, GP optimized zero-mean, var=0.4 (InvGamma(3,3)), len=0.1 (InvGamma(3,3)), noise=0.1 ∈ [0.01, 1.0] (Uniform)'
    filename = 'results/figures/benchmark/'+'accuracy_of_methods_barplot_'+str(list(metric_values.keys()))
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
                    mse_list = metric_values[splitter_key][dataset_key][algo][rep][None][metric]
                    neg_invert_mse = 1-np.mean(mse_list)
                    previous_metric = 1-np.mean(metric_values[previous_split_keys[-1]][dataset_key][algo][rep][None][metric]) if len(previous_split_keys) > 0 else 0.
                    prev_previous_metric = 1-np.mean(metric_values[previous_split_keys[-2]][dataset_key][algo][rep][None][metric]) if len(previous_split_keys) > 1 else 0.
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
                                # axs[d].arrow(previous_metric, i+seps[j], neg_invert_mse-previous_metric, 0, length_includes_head=True,
                                # color=rc.get(rep), width=height, head_width=1.5*height, head_length=0.1*(neg_invert_mse-previous_metric), ec='black')
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
                            # axs[d].arrow(previous_metric, i+seps[j], neg_invert_mse, 0, length_includes_head=True,
                            #     color=rc.get(rep), width=height, head_width=1.5*height, head_length=0.1*(neg_invert_mse-previous_metric), ec='black',
                            #     transform=Affine2D().rotate_deg_around(x=previous_metric, y=i+seps[j],degrees=180))
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
    fig.legend(handles[:len(reps)], reps, loc='lower right', prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.savefig(filename+".png")
    plt.savefig(filename+".pdf")
    plt.show()


def plot_optimization_task(metric_values: dict, name: str, max_iterations=500):
    plt.figure()
    for d, dataset_key in enumerate(metric_values.keys()):
        algos = []
        for i, algo in enumerate(metric_values[dataset_key].keys()):           
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                if algo not in algos:
                    algos.append(algo)
                observations = np.vstack(metric_values[dataset_key][algo][rep][-max_iterations:])
                means = np.mean(observations, axis=0)
                stds = np.std(observations, ddof=1, axis=0)/np.sqrt(observations.shape[0])
                plt.plot(means, color=ac.get(algo), label=algo, linewidth=2)
                plt.fill_between(list(range(len(means))), means-stds, means+stds, color=ac.get(algo), alpha=0.5)
                if 'best' in name.lower():
                    _, Y = load_dataset(dataset_key, representation=ONE_HOT)
                    plt.hlines(min(Y), 0, len(means), linestyles='--', linewidth=2.5, colors='dimgrey')
    plt.xlabel('Iterations', size=16)
    plt.ylabel('observed value', size=16)
    if 'regret' in name.lower():
        plt.ylabel('cumulative regret')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(' '.join(name.split("_")))
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in ac.values()]
    plt.legend(markers, algos, loc="lower right", numpoints=1, prop={'size':12})
    plt.tight_layout()
    plt.savefig('results/figures/optim/'+name+'_optimization_plot.png')
    plt.savefig('results/figures/optim/'+name+'_optimization_plot.pdf')
    plt.show()


def __parse_cumulative_results_dict(metrics_values: dict, metric: str, number_quantiles=10) -> dict:
    data_fractions = list(metrics_values.keys())
    dataset = list(metrics_values[data_fractions[0]].keys())[0]
    methods = list(metrics_values[data_fractions[0]][dataset].keys())
    representations = list(metrics_values[data_fractions[0]][dataset][methods[0]].keys())
    observations = {m: {'mean': [], 'std': [], 'eps': [],
                        'mean_ece': [], 'ece_err': [], 'sharpness': []} for m in methods}
    for fraction in data_fractions:
        for method in methods:
            for representation in representations:
                _results = metrics_values[fraction][dataset][method][representation][None]
                number_splits = len(list(_results.keys()))
                _metric = [_results[s][metric] for s in range(number_splits)]
                _eps = [_results[s].get(PAC_BAYES_EPS) for s in range(number_splits)]
                if metric == MSE: # compute positive 1-NMSE:
                    mean_metric = np.clip(1-np.mean(_metric), 0, 1)
                elif metric == SPEARMAN_RHO:
                    mean_metric = np.clip(np.mean(_metric), 0, 1)
                else:
                    mean_metric = np.mean(_metric)
                # TODO: STD calculated w.r.t. clipped metric?
                std_metric = np.std(_metric)/np.sqrt(len(_metric))
                observations[method]['mean'].append(mean_metric)
                observations[method]['std'].append(std_metric)
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


def cumulative_performance_plot(metrics_values: dict, metrics=[MLL, MSE, SPEARMAN_RHO], number_quantiles=10):
    data_fractions = list(metrics_values.keys())
    dataset = list(metrics_values[data_fractions[0]].keys())[0]
    methods = list(metrics_values[data_fractions[0]][dataset].keys())
    representation = list(metrics_values[data_fractions[0]][dataset][methods[0]].keys())[0]
    for metric in metrics:
        observations = __parse_cumulative_results_dict(metrics_values=metrics_values, metric=metric, number_quantiles=number_quantiles)
        fig, ax = plt.subplots(4, figsize=(10,10), gridspec_kw={'height_ratios': [2, 1, 2, 1]})
        for method in methods:
            # TODO: keep index of NaNs and annotate with stars?
            y = np.nan_to_num(np.array(observations[method]['mean']))
            ece = np.nan_to_num(np.array(observations[method]['mean_ece']))
            yerr = np.nan_to_num(np.array(observations[method]['std']))
            ece_err = np.nan_to_num(np.array(observations[method]['ece_err']))
            eps = np.nan_to_num(np.array(observations[method]['eps']))
            ax[0].errorbar(data_fractions, y, yerr=yerr, lw=3, color=ac.get(method), label=method)
            if metric == MLL:
                ax[0].set_yscale('log')
            if metric in [MSE, SPEARMAN_RHO] and method not in ['RF', 'KNN']:
                ax[0].fill_between(data_fractions, y+eps, y-eps, alpha=0.2, color=ac.get(method), label=r"PAC $\lambda$-bound $\delta$=.05")
                ax[0].set_ylim((0, 1))
            ax[1].plot(data_fractions, np.cumsum(y), marker="o", lw=3, color=ac.get(method), label=method)
            ax[2].errorbar(data_fractions, ece, yerr=ece_err, lw=3, color=ac.get(method), label=method)
            ax[3].plot(data_fractions, np.cumsum(ece), marker="o", lw=3, color=ac.get(method), label=method)
            if metric == MLL:
                ax[1].set_yscale('log')
        ax[0].set_xlabel('fraction of N')
        if metric == MSE:
            metric = "1-NMSE+"
        ax[1].set_xticks(data_fractions, np.round(data_fractions, 3), rotation=45, fontsize=12, ha='right')
        ax[3].set_xticks(data_fractions, np.round(data_fractions, 3), rotation=45, fontsize=12, ha='right')
        ax[0].set_ylabel(f'{metric}')
        ax[1].set_ylabel(f'cumulative {metric}')
        ax[2].set_ylabel('ECE')
        ax[3].set_ylabel('cumulative ECE')
        plt.suptitle(f"{dataset}: {metric} on {representation} \n over training-fractions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/figures/fraction_benchmark/{dataset}_{metric}_{representation}.png')
        plt.savefig(f'results/figures/fraction_benchmark/{dataset}_{metric}_{representation}.pdf')
        plt.show()
