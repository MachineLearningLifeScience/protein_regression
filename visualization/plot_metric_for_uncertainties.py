from cProfile import label
import numpy as np
import os
import json
from os.path import join, dirname
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import imageio
from typing import List
from shapely.geometry import Polygon
import mlflow
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from util.mlflow.constants import MSE, NO_AUGMENT, DATASET, AUGMENTATION, METHOD, REPRESENTATION, SPLIT, VAE, GP_L_VAR, OBSERVED_Y
from util.mlflow.convenience_functions import get_mlflow_results_artifacts
from uncertainty_quantification.confidence import quantile_and_oracle_errors
from uncertainty_quantification.calibration import prep_reliability_diagram, confidence_based_calibration

# MLFLOW CODE ONLY WORKS WITH THE BELOW LINE:
mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("results", "mlruns")))

def plot_confidence_curve(h_i: np.ndarray, h_o: np.ndarray, savefig=True, suffix="", metric=MSE) -> None:
    quantiles = np.arange(0, 1, 1/len(h_i))
    plt.scatter(quantiles, h_i, "r")
    plt.plot(quantiles, h_i, "r-", label="predictions")
    plt.scatter(quantiles, h_o, "k")
    plt.plot(quantiles, h_o, "k-", label="oracle")
    plt.ylabel(f"{metric}")
    plt.xlabel("quantile")
    plt.title("Confidence Curve \n quantile ranked loss")
    # TODO: add AUCO
    plt.legend()
    # TODO: savefig
    plt.show()


def plot_calibration(fractions, savefig=True, suffix="") -> None:
    quantiles = np.arange(0, 1, 1/len(fractions))
    plt.scatter(quantiles, fractions, "r")
    plt.plot(quantiles, fractions, "r-", label="calibration")
    plt.plot(np.arange(0,1), "k:", alpha=0.5)
    plt.legend()
    plt.show()


def combine_pointsets(x1,x2):
    """
    AUTHOR: Jacob KH
    Function takes two lists and combines them to a list that is 
    ready to be fed to shapely.Polygon class by outputting a new list 
    with each element being a point on the polygon. This is
    implemented by walking along x1 and then walking on x2
    and concat first point of x1. This assumes that x1 and x2 are
    on same x-axis and each element in x1 and x2 matches on the x-axis
    with x-axis [0,1] length == len(x1) == len(x2)

    Input: 
      x1: list-type of values
      x2: list-type of values
    Output:
      Shapely Polygon ready list
    """
    x0 = np.flip(np.linspace(0,1,len(x1)))
    p1 = list(zip(x0,x1))[::-1]
    p2 = list(zip(x0,x2))
    return p1+p2+[p1[0]]

def plot_polygon(ax, poly, **kwargs):
    """
    AUTHOR: Jacob KH
    Input:
      ax: matplotlib.axes, e.g fig, ax = plt.subplots()
      pgon: shapely Polygon
    Output:
      plots the input polygon
    Note: "pip install shapely"
    """
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection
    

def reliabilitydiagram(metric_values: dict, number_quantiles: int, cvtype: str = '', dataset='', representation='', optimize_flag=False, dim=None, dim_reduction=None):
    """
    Plotting calibration Curves.
    AUTHOR: Jacob KH, 
    LAST CHANGES: Richard M
    """
    c = ['dimgrey', '#661100', '#332288', 'teal']
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    algos = []
    ECE_list = []
    Sharpness_list = []
    Sharpness_std_list = []
    for d in metric_values.keys():
        for a in metric_values[d].keys():
            n_reps = len(metric_values[d][a].keys())
    fig, axs = plt.subplots(2, n_reps, figsize=(15,5), gridspec_kw={'height_ratios': [4, 1]})
    for d, dataset_key in enumerate(metric_values.keys()):
        for i, algo in enumerate(metric_values[dataset_key].keys()):           
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                for aug in metric_values[dataset_key][algo][rep].keys():
                    number_splits = len(list(metric_values[dataset_key][algo][rep][aug].keys()))
                    if algo not in algos:
                        algos.append(algo)
                    count = []
                    uncertainties_list = [] # point of investigation
                    ECE = 0
                    Sharpness = 0
                    for s in metric_values[dataset_key][algo][rep][aug].keys():
                        trues = metric_values[dataset_key][algo][rep][aug][s]['trues']
                        preds = metric_values[dataset_key][algo][rep][aug][s]['pred']
                        uncertainties = np.sqrt(metric_values[dataset_key][algo][rep][aug][s]['unc'])
                        data_uncertainties = metric_values[dataset_key][algo][rep][aug][s].get(GP_L_VAR)
                        if data_uncertainties: # in case of GP include data-noise
                            uncertainties += np.sqrt(data_uncertainties)
                        # confidence calibration
                        C, perc, E, S = prep_reliability_diagram(trues, preds, uncertainties, number_quantiles)
                        #C, perc = confidence_based_calibration(preds, uncertainties, y_ref_mean=np.mean(trues))
                        count.append(C)
                        uncertainties_list.append(np.array(uncertainties))
                        ECE += E
                        Sharpness += S / number_splits
                    count = np.mean(np.vstack(count), axis=0)
                    uncertainties = np.concatenate(uncertainties_list)
                    axs[0, j].plot(perc, count, c=c[i], lw=2, linestyle='-')
                    axs[0, j].scatter(perc, count, c=c[i], s=30-i*5)
                    axs[0, j].plot(perc,perc, ls=':', color='k', label='Perfect Calibration')
                    axs[0, j].set_title(rep)
                    axs[1, j].hist(uncertainties, 100, label=f"{algo}; {rep}", alpha=0.7, color=c[i])
                    axs[0, j].set_xlabel('percentile', size=12)
                    axs[1, j].set_xlabel('std', size=12)
                    #axs[1, j].set_xlim(0, 1.2)
                    #axs[1, j].legend()
                    ECE_list.append(ECE)
                    Sharpness_list.append(Sharpness)
    plt.legend(markers, [A+' ECE: '+str(np.round(E,3))+'\n Sharpness: '+str(np.round(S,3)) for A,E,S in zip(algos, ECE_list, Sharpness_list)], 
    loc="lower right", prop={'size':5})
    plt.suptitle(f"{str(dataset)} Calibration Split: {cvtype}, d={dim} {dim_reduction}")
    axs[0, 0].set_ylabel('Cumulative confidence', size=12)
    axs[1, 0].set_ylabel('count', size=12)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.tight_layout()
    plt.savefig(f'results/figures/uncertainties/{cvtype}_reliabilitydiagram_{dataset}_{representation}_opt_{optimize_flag}_d_{dim}{dim_reduction}.png')
    plt.show()


def multi_dim_reliabilitydiagram(metric_values: dict, number_quantiles: int, cvtype: str='', dataset='', representation='', optimize_flag=True, dim_reduction=None):
    """
    Plotting calibration Curves including results from lower dimensions.
    AUTHOR: Richard M with utility functions from JAKA H
    """
    c = ['dimgrey', '#661100', '#332288', 'teal']
    markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in c]
    algos = []
    dim = list(metric_values.keys())[0]
    data = list(metric_values[dim].keys())[0]
    alg = list(metric_values[dim][data].keys())[0]
    n_algs = len(metric_values[dim][data].keys())
    n_reps = len(metric_values[dim][data][alg].keys())
    fig, axs = plt.subplots(n_algs*2, n_reps, figsize=(17,7), gridspec_kw={'height_ratios': [4, 1]*n_algs})
    dimensions = list(metric_values.keys())[:-1] + [1128]
    shade = np.arange(0.2, 1.1, step=1/len(dimensions))
    for d_idx, (dim, _results) in enumerate(metric_values.items()):
        for d, dataset_key in enumerate(_results.keys()):
            row_idx = 0
            for i, algo in enumerate(_results[dataset_key].keys()):      
                for j, rep in enumerate(_results[dataset_key][algo].keys()):
                    for aug in _results[dataset_key][algo][rep].keys():
                        if f"d={str(dim)} {str(algo)}" not in algos:
                            algos.append(f"d={str(dim)} {str(algo)}")
                        count = []
                        uncertainties_list = [] # point of investigation
                        for s in _results[dataset_key][algo][rep][aug].keys():
                            trues = _results[dataset_key][algo][rep][aug][s]['trues']
                            preds = _results[dataset_key][algo][rep][aug][s]['pred']
                            uncertainties = np.sqrt(_results[dataset_key][algo][rep][aug][s]['unc'])
                            data_uncertainties = _results[dataset_key][algo][rep][aug][s].get(GP_L_VAR)
                            if data_uncertainties: # in case of GP include data-noise
                                uncertainties += np.sqrt(data_uncertainties)
                            # confidence calibration
                            C, perc, E, S = prep_reliability_diagram(trues, preds, uncertainties, number_quantiles)
                            count.append(C)
                            uncertainties_list.append(np.array(uncertainties))
                        count = np.mean(np.vstack(count), axis=0)
                        uncertainties = np.concatenate(uncertainties_list)
                        axs[row_idx, j].plot(perc, count, c=c[i], lw=2, linestyle='-', alpha=shade[d_idx])
                        axs[row_idx, j].plot(perc, count, color=c[i], marker="o", alpha=shade[d_idx], markersize=5+2*shade[d_idx], label=f"d={str(dim)} {str(algo)}")
                        if d_idx == 0:
                            axs[row_idx, j].plot(perc, perc, ls=':', color='k', label='Perfect Calibration')
                            axs[row_idx, 0].set_ylabel('cm. confidence', size=9)
                            axs[row_idx+1, 0].set_ylabel('count', size=9)
                        axs[row_idx, j].set_title(f"{algo} on {rep}")
                        axs[row_idx+1, j].hist(uncertainties, 100, label=f"{algo}; {rep}", alpha=shade[d_idx], color=c[i])
                        axs[row_idx, j].set_xlabel('percentile', size=12)
                        axs[row_idx+1, j].set_xlabel('std', size=12)
                row_idx += 2
    plt.legend(loc="lower right", prop={'size':5})
    plt.suptitle(f"{str(dataset)} Calibration Split: {cvtype}, {dim_reduction}")
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.tight_layout()
    plt.savefig(f'results/figures/uncertainties/dim_{str(list(metric_values.keys()))}_{cvtype}_reliabilitydiagram_{dataset}_{representation}_opt_{optimize_flag}_d_{str(dimensions)}{dim_reduction}.png')
    plt.show()


def confidence_curve(metric_values: dict, number_quantiles: int, cvtype: str = '', dataset='', representation='', 
                    optimize_flag=True, dim=None, dim_reduction=None):
    c = ['dimgrey', '#661100', '#332288']
    qs = np.linspace(0,1,1+number_quantiles)
    for d in metric_values.keys():
        n_algo = len(metric_values[d].keys())
        for a in metric_values[d].keys():
            n_reps = len(metric_values[d][a].keys())
    fig, axs = plt.subplots(n_algo, n_reps, figsize=(15,5))
    for d, dataset_key in enumerate(metric_values.keys()):
        algos = []
        for i, algo in enumerate(metric_values[dataset_key].keys()):          
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                for aug in metric_values[dataset_key][algo][rep].keys():
                    if algo not in algos:
                        algos.append(algo)
                    quantile_errs, oracle_errs = [], []
                    for s in metric_values[dataset_key][algo][rep][aug].keys():
                        uncertainties = np.sqrt(metric_values[dataset_key][algo][rep][aug][s]['unc'])
                        data_uncertainties = metric_values[dataset_key][algo][rep][aug][s].get(GP_L_VAR)
                        if data_uncertainties: # in case of GP include data-noise
                            uncertainties += np.sqrt(data_uncertainties)
                        errors = metric_values[dataset_key][algo][rep][aug][s]['mse']
                        # Ranking-based calibration
                        qe, oe = quantile_and_oracle_errors(uncertainties, errors, number_quantiles)
                        quantile_errs.append(qe)
                        oracle_errs.append(oe)

                    quantile_errs = np.mean(np.vstack(quantile_errs), 0)
                    oracle_errs = np.mean(np.vstack(oracle_errs), 0)
                    try:
                        pgon = Polygon(combine_pointsets(quantile_errs,oracle_errs)) # Assuming the OP's x,y coordinates
                    except:
                        print("ERROR building Polygon with")
                        print(f"Quantile: {quantile_errs}")
                        print(f"Oracle: {oracle_errs}")
                    
                    plot_polygon(axs[i,j], pgon, facecolor='red', edgecolor='red', alpha=0.12)
                    axs[i,j].plot(qs, np.flip(quantile_errs), lw=2, 
                    label='AUCO: '+str(np.round(pgon.area,3))+'\nError drop: '+str(np.round(quantile_errs[-1]/quantile_errs[0],3)))
                    axs[i,j].plot(qs, np.flip(oracle_errs), "k--", lw=2, label='Oracle') 
                    axs[i,j].set_ylabel('Normalized MSE', size=14)
                    axs[i,j].set_xlabel('Percentile', size=14)
                    axs[i,j].set_title(f"Rep: {rep} Algo: {algo}", size=12)
    plt.legend() 
    plt.suptitle(f"{str(dataset)} Split: {cvtype} , d={dim} {dim_reduction}")
    plt.savefig(f'results/figures/uncertainties/{algo}_{cvtype}_confidence_curve_{dataset}_{representation}_opt_{optimize_flag}_d_{dim}_{dim_reduction}.png')
    plt.tight_layout()
    plt.show()


def multi_dim_confidencecurve(metric_values: dict, number_quantiles: int, cvtype: str='', dataset='', representation='', optimize_flag=True, dim_reduction=None):
    c = ['dimgrey', '#661100', '#332288']
    qs = np.linspace(0,1,1+number_quantiles)
    dim = list(metric_values.keys())[0]
    data = list(metric_values[dim].keys())[0]
    rep = list(metric_values[dim][data].keys())[0]
    n_algo = len(metric_values[dim][data].keys())
    n_reps = len(metric_values[dim][data][rep].keys())
    fig, axs = plt.subplots(n_algo, n_reps, figsize=(15,5))
    dimensions = list(metric_values.keys())[:-1] + [1128]
    shade = np.arange(0.2, 1.1, step=1/len(dimensions))
    for d_idx, (dim, _results) in enumerate(metric_values.items()):
        for d, dataset_key in enumerate(_results.keys()):
            for i, algo in enumerate(_results[dataset_key].keys()):          
                for j, rep in enumerate(_results[dataset_key][algo].keys()):
                    for aug in _results[dataset_key][algo][rep].keys():
                        quantile_errs, oracle_errs = [], []
                        for s in _results[dataset_key][algo][rep][aug].keys():
                            uncertainties = np.sqrt(_results[dataset_key][algo][rep][aug][s]['unc'])
                            data_uncertainties = _results[dataset_key][algo][rep][aug][s].get(GP_L_VAR)
                            if data_uncertainties: # in case of GP include data-noise
                                uncertainties += np.sqrt(data_uncertainties)
                            errors = _results[dataset_key][algo][rep][aug][s]['mse']
                            # Ranking-based calibration
                            qe, oe = quantile_and_oracle_errors(uncertainties, errors, number_quantiles)
                            quantile_errs.append(qe)
                            oracle_errs.append(oe)
                        quantile_errs = np.mean(np.vstack(quantile_errs), 0)
                        oracle_errs = np.mean(np.vstack(oracle_errs), 0)
                        axs[i,j].plot(qs, np.flip(quantile_errs), lw=2, alpha=shade[d_idx], color="k",
                                    label=f'd={dim}')
                        if d_idx == 0: # TODO investigate / correct oracle errors
                            axs[i,j].plot(qs, np.flip(oracle_errs), "k--", lw=2, label='Oracle') 
                        axs[i,j].set_ylabel('NMSE', size=11)
                        axs[i,j].set_xlabel('Percentile', size=11)
                        axs[i,j].set_title(f"Rep: {rep} Algo: {algo}", size=12)
                        axs[i,j].set_ylim([0, 1.75])
    plt.legend() 
    plt.suptitle(f"{str(dataset)} Split: {cvtype} ; {dim_reduction}")
    plt.savefig(f'results/figures/uncertainties/{algo}_{cvtype}_confidence_curve_{dataset}_{representation}_opt_{optimize_flag}_d_{str(dimensions)}_{dim_reduction}.png')
    plt.tight_layout()
    plt.show()



def plot_uncertainty_eval(datasets: List[str], reps: List[str], algos: List[str], 
                        train_test_splitter,  augmentations = [NO_AUGMENT], number_quantiles: int = 10, 
                        optimize=True, d=None, dim_reduction=None, metrics=[GP_L_VAR]):
    results_dict = get_mlflow_results_artifacts(datasets=datasets, reps=reps, metrics=metrics, algos=algos, train_test_splitter=train_test_splitter, augmentation=augmentations,
                                                dim=d, dim_reduction=dim_reduction, optimize=optimize)
    confidence_curve(results_dict, number_quantiles, cvtype=train_test_splitter(datasets[-1]).get_name(), dataset=datasets[-1], representation=reps[-1], optimize_flag=optimize, dim=d, dim_reduction=dim_reduction)
    reliabilitydiagram(results_dict, number_quantiles,  cvtype=train_test_splitter(datasets[-1]).get_name(), dataset=datasets[-1], representation=reps[-1], optimize_flag=optimize, dim=d, dim_reduction=dim_reduction)


def plot_uncertainty_eval_across_dimensions(datasets: List[str], reps: List[str], algos: List[str], train_test_splitter, dimensions: List[int], augmentation = [NO_AUGMENT], number_quantiles=10,
                                            optimize=True, dim_reduction=None, metrics=[GP_L_VAR]):
    dim_results_dict = {}
    for d in dimensions:
        dim_results_dict[d] = get_mlflow_results_artifacts(datasets=datasets, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, algos=algos, augmentation=augmentation,
                                                            dim=d, dim_reduction=dim_reduction, optimize=optimize)
    multi_dim_confidencecurve(dim_results_dict, number_quantiles, cvtype=train_test_splitter(datasets[-1]).get_name(), dataset=datasets[-1], representation=reps[-1], optimize_flag=optimize, dim_reduction=dim_reduction)
    multi_dim_reliabilitydiagram(dim_results_dict, number_quantiles=number_quantiles, cvtype=train_test_splitter(datasets[-1]).get_name(), dataset=datasets[-1], representation=reps[-1], optimize_flag=optimize, dim_reduction=dim_reduction)


def plot_uncertainty_optimization(dataset: str, rep: str, seeds: List[int], algos: List[str], number_quantiles: int,
                                    min_obs_metrics: dict, regret_metrics=dict, optimize=False, stepsize=10, max_iterations=500):
    # Note: optimize is set to false, as mlflow query is different for optimization experiments vs. regression tasks, no optimize flag is set for optimization tasks
    gif_filename = f'results/figures/optim/gif/optimization_experiment_calibration_{dataset}_{rep}.gif'
    c = ['dimgrey', '#661100', '#332288']
    gif_files_list = []
    results_dict = {}
    for s in seeds:
        experiment_ids = [dataset+"_optimization"]
        results_dict[s] = get_mlflow_results_artifacts(datasets=[dataset], algos=algos, reps=[rep], seed=s, optimize=optimize, experiment_ids=experiment_ids, metrics=[OBSERVED_Y, GP_L_VAR], train_test_splitter=None)
    _recorded_algos = list(results_dict[seeds[0]][dataset].keys())
    for val_step in range(int(max_iterations/stepsize)-1):
        step = 10+val_step*stepsize
        filename = f'results/figures/optim/gif/optimization_experiment_{dataset}_{rep}_{val_step}.png'
        fig, ax = plt.subplots(3, len(algos), figsize=(15,7), gridspec_kw={'height_ratios': [4, 1, 2]})
        for k, algo in enumerate(_recorded_algos):
            count_list = []
            uncertainties_list = []
            best_observed_list = []
            regret_list = []
            # get artifacts at stepsize, make calibration plot
            for s_idx, seed in enumerate(seeds):
                _results = results_dict[seed][dataset][algo][rep][None][val_step]
                trues = _results['trues']
                preds = _results['pred']
                uncertainties = np.sqrt(_results['unc'])
                data_uncertainties = results_dict[seed][dataset][algo][rep][None][val_step].get(GP_L_VAR)
                if data_uncertainties: # in case of GP include data-noise
                    uncertainties += np.sqrt(data_uncertainties)
                # confidence calibration
                C, perc, E, S = prep_reliability_diagram(trues, preds, uncertainties, number_quantiles)
                #C, perc = confidence_based_calibration(preds, uncertainties, y_ref_mean=np.mean(trues))
                count_list.append(C)
                uncertainties_list.append(np.array(uncertainties))
                best_observed_list.append(min_obs_metrics[dataset][algo][rep][s_idx][:step])
                regret_list.append(regret_metrics[dataset][algo][rep][s_idx][:step]) # TODO
            count = np.mean(np.vstack(count_list), axis=0)
            std_count = np.std(np.vstack(count_list), axis=0)
            uncertainties = np.concatenate(uncertainties_list)
            ax[0, k].plot(perc, count, c=c[k], lw=2, linestyle='-', label=f"{algo}")
            ax[0, k].errorbar(perc, count, std_count, linestyle=None, c=c[k], marker="o", ms=7)
            ax[0, k].plot(perc, perc, ls=':', color='k', label='Perfect Calibration')
            ax[0, k].set_title(f"{algo}")
            ax[1, k].hist(uncertainties, 100, alpha=0.7, color=c[k])
            ax[1, k].set_xlim([0., 1.6])
            # ax[1, k].set_ylim([0, 750])
            ax[1, k].set_ylim([0, 1700])
            ax[0, k].set_xlabel('percentile', size=12)
            ax[1, k].set_xlabel('std', size=12)
            mean_best_observed_values = np.mean(np.vstack(best_observed_list), axis=0)
            std_observed_values = np.std(best_observed_list, ddof=1, axis=0)/np.sqrt(mean_best_observed_values.shape[0])
            mean_regret_values = np.mean(np.vstack(regret_list), axis=0)
            std_regret_values = np.std(regret_list, ddof=1, axis=0)/np.sqrt(std_observed_values.shape[0])
            ax[2, 0].plot(mean_best_observed_values, color=c[k], label=algo, linewidth=2) # best observed values over steps
            ax[2, 0].fill_between(list(range(len(mean_best_observed_values))), mean_best_observed_values-std_observed_values, 
                                            mean_best_observed_values+std_observed_values, color=c[k], alpha=0.5)
            ax[2, 1].plot(mean_regret_values, color=c[k], label=algo, linewidth=2) # regret over steps
            ax[2, 1].fill_between(list(range(len(mean_regret_values))), mean_regret_values-std_regret_values, 
                                            mean_regret_values+std_regret_values, color=c[k], alpha=0.5)
        ax[2, 0].set_xlabel('Iterations', size=16)
        ax[2, 1].set_xlabel('Iterations', size=16)
        ax[0, 0].set_ylabel('confidence', size=9)
        ax[1, 0].set_ylabel('count', size=9)
        ax[2, 0].set_ylabel('observed value', size=9)
        ax[2, 0].set_title('Best observed value')
        ax[2, 1].set_title('Regret values')
        markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in c]
        ax[2, 1].legend(markers, algos, loc="lower right", numpoints=1, prop={'size':12})
        gif_files_list.append(filename)
        plt.suptitle(f"Calibration on {rep}\n @ step {step}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    # make gif
    with imageio.get_writer(gif_filename, mode="I") as writer:
        for filename in gif_files_list:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(gif_files_list[1:-1]):
        os.remove(filename)

