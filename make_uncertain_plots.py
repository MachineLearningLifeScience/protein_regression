import os
from pickletools import optimize
import sys
from os.path import join, dirname
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy import stats
import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType
import data
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, REPRESENTATION, NO_AUGMENT, ROSETTA, TRANSFORMER, VAE, SPLIT, ONE_HOT, NONSENSE, KNN_name, VAE_DENSITY
from util.mlflow.convenience_functions import find_experiments_by_tags
from data.load_dataset import get_wildtype, get_alphabet
from visualization.plot_metric_for_dataset import barplot_metric_augmentation_comparison, barplot_metric_comparison
from typing import List

base_path = join(dirname(__file__), "results")
datasets = ["1FQG"]
train_test_splitter = RandomSplitter # BlockPostionSplitter # 
metric = MSE
last_result_length = None
reps = [TRANSFORMER, VAE, ONE_HOT]
augmentations =  [NO_AUGMENT]
number_quantiles = 10
algos = [GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), 
         UncertainRandomForest().get_name()]

def combine_pointsets(x1,x2):
    """
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
    Input:
      ax: matplotlib.axes, e.g fig, ax = plt.subplots()
      pgon: shapely Polygon
    Output:
      plots the input polygon
    Note: "pip install shapely"
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.collections import PatchCollection
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def quantile_and_oracle_errors(uncertainties, errors, number_quantiles):
    """
    Based on a list of uncertainties, a list of errors 
    and a number of quantiles this function outputs
    oracle errors that is the mean of a given percentile when
    percentiles are sorted by the errors and quantile errors that is
    the mean error when sorted by their corresponding uncertainty.

    Input:
        uncertainties: list of uncertainties
        errors: list of prediction errors
        number_quantiles: number of percentile bins between 0 and 1
                          i.e bin size reciprocal of number_quantiles
    Output:
        quantile_errs: list of mean errors when percentiles are made
                        by sorting by uncertainty
        oracle_errs: list of mean errors when percentiles are made
                        by sorting by error
    """
    quantile_errs = []
    oracle_errs = []
    qs = np.linspace(0,1,1+number_quantiles)
    s = pd.DataFrame({'unc': uncertainties, 'err': errors})
    for q in qs:
        idx = (s.sort_values(by='unc',ascending=False).reset_index(drop=True) <= s.quantile(q)).values[:,0]

        quantile_errs.append(np.mean(s.sort_values(by='unc',ascending=False).reset_index(drop=True)[idx]['err'].values))

        idx = (s.sort_values(by='err',ascending=False).reset_index(drop=True) <= s.quantile(q)).values[:,1]
        oracle_errs.append(np.mean(s.sort_values(by='err',ascending=False).reset_index(drop=True)[idx]['err'].values))

    # normalize to the case where all datapoints are included
    quantile_errs = quantile_errs/quantile_errs[-1]
    oracle_errs = oracle_errs/oracle_errs[-1]

    return quantile_errs, oracle_errs


## Uncertainty calibration
# confidence calibration
def prep_reliability_diagram(true, preds, uncertainties, number_quantiles):
    true, preds, uncertainties = np.array(true), np.array(preds), np.array(uncertainties)

    # confidence intervals
    four_sigma = 0.999936657516334
    perc = np.concatenate([np.linspace(0.1,0.9,number_quantiles-1),[four_sigma]])
    count_arr = np.vstack([np.abs(true-preds) <= stats.norm.interval(q, loc=np.zeros(len(preds)), scale=uncertainties)[1] for q in perc])
    count = np.mean(count_arr, axis=1)

    # ECE
    ECE = np.mean(np.abs(count - perc))

    # Sharpness
    Sharpness = np.std(uncertainties, ddof=1)/np.mean(uncertainties)

    return count, perc, ECE, Sharpness


def reliabilitydiagram(metric_values: dict, number_quantiles: int, cvtype: str = '', dataset='', representation='', optimize_flag=False):
    c = ['dimgrey', '#661100', '#332288']
    algos = []
    ECE_list = []
    Sharpness_list = []
    Sharpness_std_list = []
    for d in metric_values.keys():
        for a in metric_values[d].keys():
            n_reps = len(metric_values[d][a].keys())
    fig, axs = plt.subplots(1, n_reps, figsize=(15,5))
    for d, dataset_key in enumerate(metric_values.keys()):
        for i, algo in enumerate(metric_values[dataset_key].keys()):           
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                for aug in metric_values[dataset_key][algo][rep].keys():
                    number_splits = len(list(metric_values[dataset_key][algo][rep][aug].keys()))
                    if algo not in algos:
                        algos.append(algo)
                    count = []
                    ECE = 0
                    Sharpness = 0
                    for s in metric_values[dataset_key][algo][rep][aug].keys():
                        trues = metric_values[dataset_key][algo][rep][aug][s]['trues']
                        preds = metric_values[dataset_key][algo][rep][aug][s]['pred']
                        uncertainties = np.sqrt(metric_values[dataset_key][algo][rep][aug][s]['unc'])
                        
                        # confidence calibration
                        C, perc, E, S = prep_reliability_diagram(trues, preds, uncertainties, number_quantiles)
                        count.append(C)
                        ECE += E
                        Sharpness += S / number_splits
                        
                    count = np.mean(np.vstack(count), 0)
                    axs[j].plot(perc, count, c=c[i], lw=2, linestyle='-')
                    axs[j].scatter(perc, count, c=c[i], s=30-i*5)
                    axs[j].plot(perc,perc, ls=':', color='k', label='Perfect Calibration')
                    axs[j].set_title(rep)
                    ECE_list.append(ECE)
                    Sharpness_list.append(Sharpness)
    plt.suptitle(f"{str(dataset)} Calibration Split: {cvtype}")
    plt.ylabel('Cumulative confidence', size=14)
    plt.xlabel('Percentile', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c]
    plt.legend(markers, [A+' ECE: '+str(np.round(E,3))+'\n Sharpness: '+str(np.round(S,3)) for A,E,S in zip(algos, ECE_list, Sharpness_list)], 
    loc="lower right", prop={'size':12})

    plt.savefig(f'results/figures/{cvtype}_reliabilitydiagram_{dataset}_{representation}_opt_{optimize_flag}.png')
    plt.show()


def confidence_curve(metric_values: dict, number_quantiles: int, cvtype: str = '', dataset='', representation='', optimize_flag=True):
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
    plt.suptitle(f"{str(dataset)} Split: {cvtype}")
    plt.savefig(f'results/figures/{algo}_{cvtype}_confidence_curve_{dataset}_{representation}_opt_{optimize_flag}.png')
    plt.tight_layout()
    plt.show()



def plot_uncertainty_eval(datasets: List[str]=["1FQG"], reps = [VAE, TRANSFORMER, ONE_HOT],
                                        algos = [GPonRealSpace(kernel_factory= lambda:  SquaredExponential()).get_name(), 
                                        UncertainRandomForest().get_name()], train_test_splitter=train_test_splitter, 
                                        augmentations = [NO_AUGMENT], number_quantiles: int = 10, optimize=True):
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
                    exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
                    runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, max_results=1, run_view_type=ViewType.ACTIVE_ONLY)
                    assert len(runs) == 1 , rep+a+dataset+str(aug)
                    for id in runs['run_id'].to_list():
                        PATH = f"{base_path}\mlruns\{exps.experiment_id}\{id}" + "\\" + "artifacts"
                        split_dict = {}
                        for s, split in enumerate(mlflow.tracking.MlflowClient().list_artifacts(id)):
                            with open(PATH+ "\\" + split.path +'\output.json') as infile:
                                split_dict[s] = json.load(infile)
                    aug_results[aug] = split_dict
                reps_results[rep] = aug_results
            if a == 'GPsquared_exponential':
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    confidence_curve(results_dict, number_quantiles, cvtype=train_test_splitter(dataset).get_name(), dataset=dataset, representation=rep, optimize_flag=optimize)
    reliabilitydiagram(results_dict, number_quantiles,  cvtype=train_test_splitter(dataset).get_name(), dataset=dataset, representation=rep, optimize_flag=optimize)


if __name__ == "__main__":
    plot_uncertainty_eval(datasets=datasets, reps=reps,
                          algos=algos, train_test_splitter=train_test_splitter, 
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=False)