import enum
import numpy as np
import matplotlib.pyplot as plt
from data.load_dataset import load_dataset

def plot_metric_for_dataset(metric_values: dict, cvtype: str):
    c = ['darkred', 'dimgray', 'blue', 'darkorange', 'k', 'lightblue', 'green', 'purple', 'chocolate', 'red', 'lightgreen', 'indigo', 'orange', 'darkblue', 'cyan', 'olive', 'brown', 'pink']
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
            plt.errorbar(i+seps[j], mse, yerr = std, fmt='o', capsize=4, capthick=2, color=c[j], label=rep_key)
    plt.title('Accuracy of protein regression methods using '+ cvtype, size=20)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    plt.legend(markers, reps, bbox_to_anchor=(1, 1), numpoints=1, prop={'size':16})
    plt.xticks(list(range(len(metric_values.keys()))), metric_values.keys(), size=16)
    plt.yticks(size=16)
    plt.xlabel('Protein data set', size=20)
    plt.ylabel('MSE', size=20)
    plt.tight_layout()
    plt.savefig('results/figures/'+'accuracy_of_methods_'+cvtype)
    plt.show()

def barplot_metric_comparison(metric_values: dict, cvtype: str, height=0.15):
    c = ['dimgrey', '#661100', '#332288', '#117733']
    plot_heading = f'Comparison of algoritms and representations, cv-type: {cvtype}, scaled, GP optimized \n (zero-mean, var=0.4, len=0.1 ∈ [0.001, 2], noise=0.1 ∈ [0.01, 0.2] bounded),'
    filename = 'results/figures/'+'accuracy_of_methods_barplot_'+cvtype
    fig, ax = plt.subplots(1, len(metric_values.keys()), figsize=(20,5))
    axs = np.ravel(ax)
    reps = []
    for d, dataset_key in enumerate(metric_values.keys()):
        for i, algo in enumerate(metric_values[dataset_key].keys()):
            seps = np.linspace(-height*0.25*len(metric_values[dataset_key].keys()), 
                               height*0.25*len(metric_values[dataset_key].keys()), 
                               len(metric_values[dataset_key][algo].keys()))
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                if rep not in reps:
                    reps.append(rep)
                mse_list = metric_values[dataset_key][algo][rep]
                neg_invert_mse = 1-np.mean(mse_list)
                error_on_mean = np.std(mse_list, ddof=1)/np.sqrt(len(mse_list))
                axs[d].barh(i+seps[j], neg_invert_mse, xerr=error_on_mean, height=height, label=rep, color=c[j], 
                            facecolor=c[j], edgecolor=c[j], ecolor='black', capsize=5, hatch='//')
                axs[d].axvline(0, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
                axs[d].axvline(-1, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
                axs[d].axvline(0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
                axs[d].axvline(-0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
                axs[d].set_yticks(list(range(len(list(metric_values[dataset_key].keys())))))
                axs[d].set_yticklabels(['' for i in range(len(list(metric_values[dataset_key].keys())))])
                axs[0].set_yticklabels(list(metric_values[dataset_key].keys()), size=16)
                axs[d].set_xlim((-1, 1))
                axs[d].tick_params(axis='x', which='both', labelsize=14)
                axs[d].set_title(dataset_key, size=16)
                axs[d].set_xlabel('1 minus normalized MSE', size=14)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:len(reps)], reps, loc='lower right', prop={'size': 14})
    plt.suptitle(plot_heading, size=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def barplot_metric_augmentation_comparison(metric_values: dict, cvtype: str, augmentation: dict, height=0.3):
    c = ['dimgrey', '#661100', '#332288', '#117733']
    cc = ['cyan', 'darkorange', 'deeppink', 'royalblue']
    augmentations_string = " ".join(augmentation)
    plot_heading = f'Augmented models and representations, cv-type: {cvtype}, augmentation {augmentations_string}, OPTIMIZED'
    filename = 'results/figures/'+'accuracy_of_methods_barplot_' + cvtype + "_".join(augmentation)
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
                    mse_list = metric_values[dataset_key][algo][rep][aug]
                    neg_invert_mse = 1-np.mean(mse_list)
                    error_on_mean = np.std(mse_list, ddof=1)/np.sqrt(len(mse_list))
                    axs[i].barh(j+seps[idx], neg_invert_mse, xerr=error_on_mean, height=height*0.25, label=repname, color=c[k], 
                            facecolor=c[k], edgecolor=cc[l], ecolor='black', capsize=5, hatch='/', linewidth=2)
                    axs[i].axvline(0, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
                    axs[i].axvline(-1, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.5)
                    axs[i].axvline(0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
                    axs[i].axvline(-0.5, seps[0], len(metric_values[dataset_key].keys())-1+seps[-1], c='grey', ls='--', alpha=0.25)
                    axs[i].set_yticks(list(range(len(list(metric_values[dataset_key].keys())))))
                    axs[i].set_yticklabels(['' for i in range(len(list(metric_values[dataset_key].keys())))])
                    axs[0].set_yticklabels(list(metric_values[dataset_key].keys()), size=16)
                    axs[i].set_xlim((-1, 0.75))
                    axs[i].tick_params(axis='x', which='both', labelsize=14)
                    axs[i].set_title(dataset_key, size=16)
                    axs[i].set_xlabel('1 minus normalized MSE', size=14)
                    idx += 1
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:len(representations)], representations, loc='lower right', prop={'size': 14})
    plt.suptitle(plot_heading, size=20)
    plt.savefig(filename)
    plt.tight_layout()
    plt.show()


def plot_optimization_task(metric_values: dict, name: str, max_iterations=500):
    c = ['dimgrey', '#661100', '#332288']
    plt.figure()
    for d, dataset_key in enumerate(metric_values.keys()):
        algos = []
        for i, algo in enumerate(metric_values[dataset_key].keys()):           
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                if algo not in algos:
                    algos.append(algo)
                # TODO: the way these results are collected is very messy.
                # TODO each experiment should be distinct and partial runs not be recorded, all should have the same len
                observations = np.vstack(metric_values[dataset_key][algo][rep][-max_iterations:])
                means = np.mean(observations, axis=0)
                stds = np.std(observations, ddof=1, axis=0)/np.sqrt(observations.shape[0])
                plt.plot(means, color=c[i], label=algo, linewidth=2)
                plt.fill_between(list(range(len(means))), means-stds, means+stds, color=c[i], alpha=0.5)
                if name == 'Best observed':
                    _, Y = load_dataset(dataset_key, representation=rep)
                    plt.hlines(min(Y), 0, len(means), linestyles='--', colors='dimgrey')

    plt.xlabel('Iterations', size=16)
    plt.ylabel(name, size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    plt.legend(markers, algos, loc="lower right", numpoints=1, prop={'size':12})
    plt.tight_layout()
    plt.savefig('results/figures/'+name+'_optimization_plot')
    plt.show()