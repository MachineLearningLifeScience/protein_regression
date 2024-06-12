"""
Utility functions for loading and processing MlFlow outputs.

NOTE: 
String based searches are very slow, thus loading and processing is very slow.
This can be done more efficiently by loading and subselecting based on query DataFrames.
Refactor of this module has to be done (TODO).
"""

import json
import os
import pickle
import re
from os.path import join
from typing import List, Tuple

import mlflow
import numpy as np
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

from data.train_test_split import AbstractTrainTestSplitter
from util.mlflow.constants import (
    AUGMENTATION,
    DATASET,
    DIMENSION,
    EVE,
    LINEAR,
    METHOD,
    NO_AUGMENT,
    NON_LINEAR,
    OBSERVED_Y,
    ONE_HOT,
    OPTIMIZATION,
    REPRESENTATION,
    SEED,
    SPLIT,
    STD_Y,
    THRESHOLD,
    VAE,
)

mlflow.set_tracking_uri("file:" + join(os.getcwd(), join("results", "mlruns")))


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
            result_list[i] += diff * [
                np.nan
            ]  # add as many nan values as are missing at that position
    return result_list, correct


def get_mlflow_results(
    datasets: list,
    algos: list,
    reps: list,
    metrics: list,
    train_test_splitter: AbstractTrainTestSplitter,
    augmentation: List[str] = [None],
    dim: int = None,
    dim_reduction: str = None,
    seed: int = None,
    optimized=True,
    artifacts=False,
    experiment_ids=None,
    threshold: List[float] = [None],
) -> dict:
    experiment_ids = datasets if not experiment_ids else experiment_ids
    results_dict = {}
    for i, dataset in enumerate(datasets):
        exps = mlflow.tracking.MlflowClient().get_experiment_by_name(experiment_ids[i])
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                aug_results = {}
                for aug in augmentation:
                    filter_string = (
                        f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}'"
                    )
                    if rep:
                        filter_string += f" and tags.{REPRESENTATION} = '{rep}'"
                    if train_test_splitter:
                        filter_string += (
                            f" and tags.{SPLIT} = '{train_test_splitter.get_name()}'"
                        )
                    if aug:
                        filter_string += f" and tags.{AUGMENTATION} = '{aug}'"
                    if (
                        dim
                        and not (rep == VAE and dim >= 30)
                        and not (rep == EVE and dim >= 50)
                    ):
                        filter_string += f" and tags.{DIMENSION} = '{dim}' and tags.DIM_REDUCTION = '{dim_reduction}'"
                    if seed:
                        filter_string += f" and tags.{SEED} = '{seed}'"
                    if threshold[0] and threshold[i]:
                        filter_string += f" and tags.{THRESHOLD} = '{threshold}'"
                    if not "_optimization" in exps.name:
                        filter_string += f" and tags.OPTIMIZE = '{str(optimized)}'"
                    runs = mlflow.search_runs(
                        experiment_ids=[exps.experiment_id],
                        filter_string=filter_string,
                        run_view_type=ViewType.ACTIVE_ONLY,
                    )
                    runs = runs[runs["status"] == "FINISHED"]
                    if not dim and f"tags.{DIMENSION}" in runs.columns:
                        runs = runs[runs[f"tags.{DIMENSION}"].isnull()]
                    if not aug and f"tags.{AUGMENTATION}" in runs.columns:
                        runs = runs[runs[f"tags.{AUGMENTATION}"] == str(aug)]
                    if not threshold[0] and f"tags.{THRESHOLD}" in runs.columns:
                        runs = runs[
                            (runs[f"tags.{THRESHOLD}"].isnull())
                            | (runs["tags.THRESHOLD"] == "None")
                        ]
                    # DEFAULT EVE case: no reduction
                    if len(runs) == 0 and rep == EVE and (dim is None or dim > 50):
                        filter_string = filter_string.replace(
                            f"and tags.{DIMENSION} = '{dim}' and tags.DIM_REDUCTION = '{dim_reduction}'",
                            "",
                        )
                        runs = mlflow.search_runs(
                            experiment_ids=[exps.experiment_id],
                            filter_string=filter_string,
                            max_results=1,
                            run_view_type=ViewType.ACTIVE_ONLY,
                        )
                    if len(runs) == 0 and dim is not None:  # dimensions lower
                        _dim = dim
                        while (
                            len(runs) == 0 and dim and _dim >= 1
                        ):  # for lower-dimensions, if not exists: take next smaller in steps of 10, e.g. one-hot reduction d=1000 may not compute
                            _dim = int(
                                re.search(r"\'\d+\'", filter_string).group()[1:-1]
                            )  # NOTE: \' to cover if other elements in the string have int elements
                            lower_dim = _dim - int(dim / 10)
                            filter_string = filter_string.replace(
                                f"tags.{DIMENSION} = '{_dim}'",
                                f"tags.{DIMENSION} = '{lower_dim}'",
                            )
                            runs = mlflow.search_runs(
                                experiment_ids=[exps.experiment_id],
                                filter_string=filter_string,
                                max_results=1,
                                run_view_type=ViewType.ACTIVE_ONLY,
                            )
                    runs = runs.iloc[:1]  # pick latest run, by implicit results order
                    assert len(runs) == 1, (
                        str(rep) + str(a) + str(dataset) + str(augmentation)
                    )
                    metric_results = {metric: [] for metric in metrics}
                    for metric in metrics:
                        try:
                            for r in mlflow.tracking.MlflowClient().get_metric_history(
                                runs.run_id.values[0], metric
                            ):
                                metric_results[metric].append(r.value)
                        except MlflowException as e:
                            continue
                    if dim is not None:  # Save reduced dimension for later plotting
                        if re.search(r"\'\d+\'", filter_string) == None:
                            metric_results["dim"] = "full"
                        else:
                            metric_results["dim"] = int(
                                re.search(r"\'\d+\'", filter_string).group()[1:-1]
                            )
                    aug_results[aug] = metric_results
                reps_results[rep] = aug_results
            if a == "GPsquared_exponential":  # NOTE: backward compatibility
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    return results_dict


# TODO: Refactor mlflow functions such that there is a string builder for queries, a query function which returns experiment and a artifact function which uses these experiments/runs
def get_mlflow_results_artifacts(
    datasets: list,
    algos: list,
    reps: list,
    metrics: list,
    train_test_splitter: AbstractTrainTestSplitter,
    augmentation: list = [None],
    dim: int = None,
    dim_reduction: str = NON_LINEAR,
    seed: int = None,
    optimize: bool = True,
    experiment_ids=None,
    threshold: List[float] = [None],
) -> dict:
    experiment_ids = datasets if not experiment_ids else experiment_ids
    results_dict = {}
    for i, dataset in enumerate(datasets):
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                aug_results = {}
                for aug in augmentation:
                    filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}'"
                    if train_test_splitter:
                        filter_string += (
                            f" and tags.{SPLIT} = '{train_test_splitter.get_name()}'"
                        )
                    if "GP" in a and not "optimization" in experiment_ids[0]:
                        filter_string += f" and tags.OPTIMIZE = '{optimize}'"
                    if (
                        dim
                        and not (rep == VAE and dim >= 30)
                        and not (rep == EVE and dim >= 50)
                    ):  # default results for VAE, EVE is highest dim
                        filter_string += f" and tags.{DIMENSION} = '{dim}' and tags.DIM_REDUCTION = '{dim_reduction}'"
                    if aug:
                        filter_string += f" and tags.{AUGMENTATION} = '{aug}'"
                    if seed:
                        filter_string += f" and tags.{SEED} = '{seed}'"
                    if threshold[0] and threshold[i]:
                        filter_string += f" and tags.{THRESHOLD} = '{threshold}'"
                    # if optimize:
                    #     filter_string += f" and tags.OPTIMIZE = 'True'"
                    exps = mlflow.tracking.MlflowClient().get_experiment_by_name(
                        experiment_ids[i]
                    )
                    runs = mlflow.search_runs(
                        experiment_ids=[exps.experiment_id],
                        filter_string=filter_string,
                        run_view_type=ViewType.ACTIVE_ONLY,
                    )
                    runs = runs[runs["status"] == "FINISHED"]
                    if not dim and f"tags.{DIMENSION}" in runs.columns:
                        runs = runs[runs["tags.DIM"].isnull()]
                    if not aug and f"tags.{AUGMENTATION}" in runs.columns:
                        runs = runs[runs["tags.AUGMENTATION"] == str(aug)]
                    # if not threshold[0] and f'tags.{THRESHOLD}' in runs.columns: # TODO: test this!
                    #     runs = runs[runs[f'tags.{THRESHOLD}'].isnull()]
                    # refine search, as query string does not allow for dim=None and we need very specific run
                    runs = runs.iloc[:1]  # get most recent result
                    try:
                        assert len(runs) == 1, rep + a + dataset + str(aug)
                    except:
                        print(f"Missing run: {rep} {a} {dataset} {train_test_splitter}")
                        continue  # TODO FIXME
                    for id in runs["run_id"].to_list():
                        PATH = (
                            f"/Users/rcml/protein_regression/results/mlruns/{exps.experiment_id}/{id}"
                            + "/"
                            + "artifacts"
                        )  # TODO: replace with project path
                        split_dict = {}
                        for s, split in enumerate(
                            mlflow.tracking.MlflowClient().list_artifacts(id)
                        ):
                            with open(
                                PATH + "//" + split.path + "/output.json"
                            ) as infile:
                                split_dict[s] = json.load(infile)
                        for metric in metrics:
                            try:  # NOTE: artifacts are NOT always in alignment with metric, e.g. 500 observations BUT only 50 artifacts
                                for s, r in enumerate(
                                    mlflow.tracking.MlflowClient().get_metric_history(
                                        id, metric
                                    )
                                ):
                                    if not s in split_dict.keys():
                                        split_dict[s] = {}
                                    split_dict[s][metric] = r.value
                            except MlflowException as e:
                                continue
                    aug_results[aug] = split_dict
                reps_results[rep] = aug_results
            if a == "GPsquared_exponential":
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    return results_dict


def get_mlflow_results_optimization(
    datasets: list,
    algos: list,
    reps: list,
    metrics: list,
    augmentation: list = [None],
    dim: int = None,
    dim_reduction: str = None,
    seeds: List[int] = [None],
) -> dict:
    experiment_ids = [d + "_optimization" for d in datasets]
    results_dict = {
        seed: get_mlflow_results(
            datasets=datasets,
            algos=algos,
            reps=reps,
            metrics=metrics,
            train_test_splitter=None,
            augmentation=augmentation,
            dim=dim,
            dim_reduction=dim_reduction,
            seed=seed,
            experiment_ids=experiment_ids,
        )
        for seed in seeds
    }  # in reference case / or if seed unset -> key=None
    return results_dict


def aggregate_fractional_splitter_results(
    split_results: list, metrics: list, augmentation=None
) -> dict:
    """
    Collect all fractional spliter results: for each split concatenate results in 2D array s.t.
    axis 0: is CV results on fraction
    axis 1: concatenates MSE results at fractions, s.t. first mse is first split, etc.
    """
    results_dict = {}
    metric_results = {}
    for split in split_results:
        datasets = list(split.keys())
        for dataset in split:
            for method in split.get(dataset):
                methods = list(split.get(dataset).keys())
                for representation in split.get(dataset).get(method):
                    representations = list(split.get(dataset).get(method).keys())
                    if dataset not in metric_results.keys():
                        metric_results[dataset] = {}
                    for metric in metrics:
                        if metric not in metric_results[dataset].keys():
                            metric_results[dataset][metric] = []
                        metric_results[dataset][metric].append(
                            np.array(
                                split.get(dataset)
                                .get(method)
                                .get(representation)
                                .get(augmentation)
                                .get(metric)
                            )
                        )
    results_dict = stack_and_parse_into_dict(
        metric_results,
        datasets=datasets,
        methods=methods,
        representations=representations,
        metrics=metrics,
    )
    return results_dict


def aggregate_optimization_results(
    opt_results: list,
    metrics: list,
    augmentation=None,
) -> dict:
    """
    Collect all optimization results across seeds: for each split concatenate results in 2D array s.t.
    axis 0: is seeds results on fraction
    axis 1: concatenates MSE results at fractions, s.t. first mse is first step, etc.
    """
    datasets = []
    methods = []
    representations = []
    metric_results = {}
    for seed in opt_results:  # drop seeds from results_dict
        for dataset in opt_results.get(seed):
            if dataset not in datasets:
                datasets.append(dataset)
            if dataset not in metric_results.keys():
                metric_results[dataset] = {}
            for method in opt_results.get(seed).get(dataset):
                if method not in methods:
                    methods.append(method)
                for representation in opt_results.get(seed).get(dataset).get(method):
                    if representation not in representations:
                        representations.append(representation)
                    for metric in metrics:
                        if metric not in metric_results[dataset].keys():
                            metric_results[dataset][metric] = []
                        metric_results[dataset][metric].append(
                            np.array(
                                opt_results.get(seed)
                                .get(dataset)
                                .get(method)
                                .get(representation)
                                .get(augmentation)
                                .get(metric)
                            )
                        )
    results_dict = stack_and_parse_into_dict(
        metric_results,
        datasets=datasets,
        methods=methods,
        representations=representations,
        metrics=metrics,
    )
    return results_dict


def stack_and_parse_into_dict(
    metric_results: dict,
    datasets: list,
    methods: list,
    representations: list,
    metrics: list,
    augmentation=None,
) -> dict:
    results_dict = {}
    # stack observations 2D:
    for dataset in datasets:
        for metric in metrics:
            metric_results[dataset][metric] = np.vstack(metric_results[dataset][metric])
    # put together results:
    for dataset in datasets:
        results_dict[dataset] = {}
        for method in methods:
            for representation in representations:
                for metric in metrics:
                    results_dict[dataset] = {
                        method: {
                            representation: {augmentation: metric_results.get(dataset)}
                        }
                    }
    return results_dict


def load_results_dict_from_mlflow(
    datasets: List[str],
    algos: List[str],
    metrics: List[str],
    reps: List[str],
    train_test_splitter: list,
    seeds: List[int] = None,
    cache=False,
    cache_fname=None,
    optimized=True,
    dim=None,
    dim_reduction=LINEAR,
) -> dict:
    results_dict = {}
    fractional_splitter_results = []
    optimization_dict = None
    for splitter in train_test_splitter:
        if "optimization" in splitter.get_name().lower():
            optimization_dict = get_mlflow_results_optimization(
                datasets=datasets,
                algos=algos,
                reps=reps,
                metrics=metrics + [OBSERVED_Y, STD_Y],
                seeds=seeds,
            )
        elif "fraction" in splitter.get_name().lower():
            _dict = get_mlflow_results(
                datasets=datasets,
                algos=algos,
                reps=reps,
                metrics=metrics,
                train_test_splitter=splitter,
                augmentation=[None],
                dim=dim,
                dim_reduction=dim_reduction,
            )
            fractional_splitter_results.append(_dict)
        else:
            _dict = get_mlflow_results(
                datasets=datasets,
                algos=algos,
                reps=reps,
                metrics=metrics,
                train_test_splitter=splitter,
                optimized=optimized,
                augmentation=[None],
                dim=dim,
                dim_reduction=dim_reduction,
            )
            results_dict[splitter.get_name()] = _dict
    if fractional_splitter_results:
        _fractional_results = aggregate_fractional_splitter_results(
            fractional_splitter_results, metrics=metrics
        )
        results_dict["Fractional"] = _fractional_results
    if optimization_dict:
        _opt_results = aggregate_optimization_results(
            optimization_dict, metrics=metrics
        )
        results_dict["Optimization"] = _opt_results
    if cache and cache_fname:
        with open(cache_fname, "wb") as outfile:
            pickle.dump(results_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return results_dict


def load_cached_results(cached_results_filename: str) -> dict:
    with open(cached_results_filename, "rb") as infile:
        results_dict = pickle.load(infile)
    return results_dict
