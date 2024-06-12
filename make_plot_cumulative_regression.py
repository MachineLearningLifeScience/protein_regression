import os
import pickle
from typing import List

import numpy as np
from gpflow.kernels import Matern52

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.uncertain_rf import UncertainRandomForest
from data.train_test_split import FractionalRandomSplitter
from util.mlflow.constants import (
    ESM,
    EVE,
    GP_L_VAR,
    LINEAR,
    MSE,
    ONE_HOT,
    STD_Y,
    TRANSFORMER,
)
from util.mlflow.convenience_functions import get_mlflow_results_artifacts
from visualization.plot_metric_for_dataset import cumulative_performance_plot


def plot_cumulative_comparison(
    datasets: List[str],
    algos: List[str],
    metrics: str,
    reps: List[str],
    testing_fractions,
    train_test_splitters,
    dimension=None,
    dim_reduction=None,
    threshold: float = None,
    cached_results: bool = False,
    savefig=True,
):
    cached_filename = f"/Users/rcml/protein_regression/results/cache/results_cumulative_split_d={'_'.join(datasets)}_a={'_'.join(algos)}_r={'_'.join(reps)}_m={'_'.join(metrics)}_s={'_'.join([s.get_name()[:5] for s in train_test_splitters[:5]])}.pkl"
    if cached_results and os.path.exists(cached_filename):
        print(f"Loading cached results: {cached_filename}")
        with open(cached_filename, "rb") as infile:
            results_dict = pickle.load(infile)
    else:
        results_dict = {}
        for frac, splitter in zip(testing_fractions, train_test_splitters):
            splitter_dict = get_mlflow_results_artifacts(
                datasets=datasets,
                algos=algos,
                reps=reps,
                metrics=metrics,
                train_test_splitter=splitter,
                dim=dimension,
                dim_reduction=dim_reduction,
                threshold=threshold,
            )
            results_dict[frac] = splitter_dict
        if cached_results:
            with open(cached_filename, "wb") as outfile:
                pickle.dump(results_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    if len(datasets) > 1:
        for dataset in datasets:
            _results_dict = {}
            for split in results_dict:
                _results_dict[split] = {}
                _results_dict[split][dataset] = results_dict[split][
                    dataset
                ]  # TODO: subset w.r.t. one data-set
            cumulative_performance_plot(
                _results_dict, threshold=threshold, savefig=savefig, metrics=metrics
            )
    else:
        cumulative_performance_plot(
            results_dict, threshold=threshold, savefig=savefig, metrics=metrics
        )


if __name__ == "__main__":
    datasets = [
        "1FQG",
        "CALM",
        "BRCA",
        "UBQT",
        "MTH3",
        "TIMB",
    ]  # ["TOXI"] # "MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"
    thresholds = [None]
    algos = [
        GPonRealSpace().get_name(),
        GPonRealSpace(kernel_factory=lambda: Matern52()).get_name(),
        UncertainRandomForest().get_name(),
    ]
    testing_fractions = np.concatenate(
        [
            np.arange(0.001, 0.3, 0.01),
            np.arange(0.3, 0.6, 0.03),
            np.arange(0.6, 1.05, 0.05),
        ]
    )
    train_test_splitters = [
        FractionalRandomSplitter(datasets[0], n) for n in testing_fractions
    ]
    metrics = [MSE, GP_L_VAR, STD_Y]
    representations = [TRANSFORMER, ESM, ONE_HOT, EVE]
    dim = None
    dim_reduction = LINEAR  # LINEAR, NON_LINEAR
    # individual dataset
    plot_cumulative_comparison(
        datasets=["1FQG"],
        algos=algos,
        metrics=metrics,
        reps=representations,
        testing_fractions=testing_fractions,
        train_test_splitters=train_test_splitters,
        dimension=dim,
        dim_reduction=dim_reduction,
        threshold=thresholds,
        cached_results=True,
    )
    plot_cumulative_comparison(
        datasets=["UBQT"],
        algos=algos,
        metrics=metrics,
        reps=representations,
        testing_fractions=testing_fractions,
        train_test_splitters=train_test_splitters,
        dimension=dim,
        dim_reduction=dim_reduction,
        threshold=thresholds,
        cached_results=True,
    )
    plot_cumulative_comparison(
        datasets=["TIMB"],
        algos=algos,
        metrics=metrics,
        reps=representations,
        testing_fractions=testing_fractions,
        train_test_splitters=train_test_splitters,
        dimension=dim,
        dim_reduction=dim_reduction,
        threshold=thresholds,
        cached_results=True,
    )
