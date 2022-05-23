from cProfile import label
import numpy as np
import os
import json
from os.path import join, dirname
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from typing import List
from shapely.geometry import Polygon
import mlflow
from mlflow.entities import ViewType
from util.mlflow.constants import MSE, NO_AUGMENT, DATASET, AUGMENTATION, METHOD, REPRESENTATION, SPLIT, VAE
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
    c = ['dimgrey', '#661100', '#332288']
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
    plt.savefig(f'results/figures/{cvtype}_reliabilitydiagram_{dataset}_{representation}_opt_{optimize_flag}_d_{dim}{dim_reduction}.png')
    plt.show()


def confidence_curve(metric_values: dict, number_quantiles: int, cvtype: str = '', dataset='', representation='', optimize_flag=True, dim=None, dim_reduction=None):
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
    plt.savefig(f'results/figures/{algo}_{cvtype}_confidence_curve_{dataset}_{representation}_opt_{optimize_flag}_d_{dim}_{dim_reduction}.png')
    plt.tight_layout()
    plt.show()


def plot_uncertainty_eval(datasets: List[str], reps: List[str], algos: List[str], 
                        train_test_splitter,  augmentations = [NO_AUGMENT], number_quantiles: int = 10, optimize=True, d=None, dim_reduction=None):
    results_dict = {}
    for dataset in datasets:
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                aug_results = {}
                for aug in augmentations:
                    filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}' and tags.{SPLIT} = '{train_test_splitter(dataset).get_name()}' and tags.{AUGMENTATION} = '{aug}'"
                    if 'GP' in a:
                        filter_string += f" and tags.OPTIMIZE = '{optimize}'"
                    if d and not (rep==VAE and d >= 30):
                        filter_string += f" and tags.DIM = '{d}' and tags.DIM_REDUCTION = '{dim_reduction}'"
                    exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
                    runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, max_results=1, run_view_type=ViewType.ACTIVE_ONLY)
                    assert len(runs) == 1 , rep+a+dataset+str(aug)
                    for id in runs['run_id'].to_list():
                        PATH = f"/Users/rcml/protein_regression/results/mlruns/{exps.experiment_id}/{id}" + "/" + "artifacts"
                        split_dict = {}
                        for s, split in enumerate(mlflow.tracking.MlflowClient().list_artifacts(id)):
                            with open(PATH+ "//" + split.path +'/output.json') as infile:
                                split_dict[s] = json.load(infile)
                    aug_results[aug] = split_dict
                reps_results[rep] = aug_results
            if a == 'GPsquared_exponential':
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    #confidence_curve(results_dict, number_quantiles, cvtype=train_test_splitter(dataset).get_name(), dataset=dataset, representation=rep, optimize_flag=optimize, dim=d, dim_reduction=dim_reduction)
    reliabilitydiagram(results_dict, number_quantiles,  cvtype=train_test_splitter(dataset).get_name(), dataset=dataset, representation=rep, optimize_flag=optimize, dim=d, dim_reduction=dim_reduction)

