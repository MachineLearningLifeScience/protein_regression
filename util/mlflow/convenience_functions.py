import re
import os
import json
import warnings
from os.path import join
import numpy as np
import mlflow
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from util.mlflow.constants import DATASET, METHOD, REPRESENTATION, SPLIT, SEED
from util.mlflow.constants import AUGMENTATION, NO_AUGMENT, DIMENSION, LINEAR, NON_LINEAR, VAE
from data.train_test_split import AbstractTrainTestSplitter
from typing import List, Tuple

mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("results", "mlruns")))


def make_experiment_name_from_tags(tags: dict) -> str:
    return "".join([t + "_" + tags[t] + "__" for t in tags.keys()])


def find_experiments_by_tags(tags: dict) -> list:
    exps = mlflow.tracking.MlflowClient().list_experiments()
    print(exps)
    def all_tags_match(e):
        for tag in tags.keys():
            if tag not in e.tags:
                return False
            if e.tags[tag] != tags[tag]:
                return False
        return True
    return [e for e in exps if all_tags_match(e)]


def check_results(result_list: list, fill_with_na=True) -> Tuple[list, bool]:
    """
    Utility function that ensures that outputs are of the same dimensionality, in case of missing values.
    If fill_with_na (default=True), then fill missing positions with np.nan values.
    Returns list of results and flag if all elements in list are of the same length.
    """
    correct = True
    max_len = np.max([len(e) for e in result_list])
    _result_list = result_list.copy()
    for i, e in enumerate(_result_list):
        if len(e) < max_len:
            correct = False
        if fill_with_na:
            diff = max_len - len(e)
            result_list[i] += diff*[np.nan] # add as many nan values as are missing at that position
    return result_list, correct





def get_mlflow_results(datasets: list, algos: list, reps: list, metrics: list, train_test_splitter: AbstractTrainTestSplitter, 
                    augmentation: list=[None], dim=None, dim_reduction=NON_LINEAR, seed=None, artifacts=False) -> dict:
    results_dict = {}
    for dataset in datasets:
        exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                aug_results = {}
                for aug in augmentation:
                    filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}'"
                    if train_test_splitter:
                        filter_string += f" and tags.{SPLIT} = '{train_test_splitter(dataset).get_name()}'"
                    if aug:
                        filter_string += f" and tags.{AUGMENTATION} = '{aug}'"
                    if dim and not (rep==VAE and dim >= 30):
                        filter_string += f" and tags.{DIMENSION} = '{dim}' and tags.DIM_REDUCTION = '{dim_reduction}'"
                    if seed:
                        filter_string += f" and tags.{SEED} = '{seed}'"
                    runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, max_results=1, run_view_type=ViewType.ACTIVE_ONLY)
                    while len(runs) != 1 and dim and dim >= 1: # for lower-dimensional experiments, if not exists: take next smaller in steps of 10:
                        _dim = int(re.search(r'\d+', filter_string).group())
                        lower_dim = _dim - int(dim/10)
                        filter_string = filter_string.replace(f"tags.{DIMENSION} = '{_dim}'", f"tags.{DIMENSION} = '{lower_dim}'")
                        runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, max_results=1, run_view_type=ViewType.ACTIVE_ONLY)
                    assert len(runs) == 1 , rep+a+dataset+str(augmentation)
                    metric_results = {metric: [] for metric in metrics}     
                    for id in runs['run_id'].to_list():
                        for metric in metrics:
                            try:
                                for r in mlflow.tracking.MlflowClient().get_metric_history(id, metric):
                                    metric_results[metric].append(r.value)
                            except MlflowException as e:
                                continue
                    aug_results[aug] = metric_results
                reps_results[rep] = aug_results
            if a == 'GPsquared_exponential':
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    return results_dict


# TODO: Refactor mlflow functions such that there is a string builder for queries, a query function which returns experiment and a artifact function which uses these experiments/runs
def get_mlflow_results_artifacts(datasets: list, algos: list, reps: list, metrics: list, train_test_splitter: AbstractTrainTestSplitter, 
                    augmentation: list=[None], dim=None, dim_reduction=NON_LINEAR, seed=None, optimize=True) -> dict:
    results_dict = {}
    for dataset in datasets:
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                aug_results = {}
                for aug in augmentation:
                    filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}' and tags.{SPLIT} = '{train_test_splitter(dataset).get_name()}' and tags.{AUGMENTATION} = '{aug}'"
                    if 'GP' in a:
                        filter_string += f" and tags.OPTIMIZE = '{optimize}'"
                    if dim and not (rep==VAE and dim >= 30):
                        filter_string += f" and tags.DIM = '{dim}' and tags.DIM_REDUCTION = '{dim_reduction}'"
                    exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
                    runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, max_results=1, run_view_type=ViewType.ACTIVE_ONLY)
                    assert len(runs) == 1 , rep+a+dataset+str(aug)
                    for id in runs['run_id'].to_list():
                        PATH = f"/Users/rcml/protein_regression/results/mlruns/{exps.experiment_id}/{id}" + "/" + "artifacts"
                        split_dict = {}
                        for s, split in enumerate(mlflow.tracking.MlflowClient().list_artifacts(id)):
                            with open(PATH+ "//" + split.path +'/output.json') as infile:
                                split_dict[s] = json.load(infile)
                        for metric in metrics:
                            try:
                                for s, r in enumerate(mlflow.tracking.MlflowClient().get_metric_history(id, metric)):
                                    split_dict[s][metric] = r.value
                            except MlflowException as e:
                                continue
                    aug_results[aug] = split_dict
                reps_results[rep] = aug_results
            if a == 'GPsquared_exponential':
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    return results_dict


def get_mlflow_results_optimization(datasets: list, algos: list, reps: list, metrics: list, augmentation: list=None, dim: int=None, dim_reduction: str=None, seeds: List[int]=None):
    if seeds:
            results_dict = {seed: get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, 
                                                    train_test_splitter=None, augmentation=augmentation, dim=dim, dim_reduction=dim_reduction, seed=seed) 
                            for seed in seeds}
    else:
        results_dict = get_mlflow_results(datasets=datasets, algos=algos, metrics=metrics, train_test_splitter=None, augmentation=augmentation, dim=dim, dim_reduction=dim_reduction)
    return results_dict
