import sys
from os.path import exists
import pickle
import numpy as np
from scipy.special import comb
from gpflow.kernels import SquaredExponential, Matern52
from data.train_test_split import PositionSplitter, RandomSplitter, BioSplitter, AbstractTrainTestSplitter, WeightedTaskSplitter, FractionalRandomSplitter, OptimizationSplitter
from protocol_factories import FractionalSplitterFactory
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, LINEAR, METHOD, MSE, SPEARMAN_RHO
from util.mlflow.constants import NON_LINEAR, REPRESENTATION, ROSETTA, TRANSFORMER, VAE, ESM, VAE_AUX, VAE_RAND, EVE, EVE_DENSITY
from util.mlflow.constants import SPLIT, ONE_HOT, NONSENSE, KNN_name, VAE_DENSITY, VAE_AUX, NO_AUGMENT, LINEAR, NON_LINEAR, MEAN_Y, STD_Y, OBSERVED_Y
from util.mlflow.convenience_functions import find_experiments_by_tags, get_mlflow_results, get_mlflow_results_optimization
from util.mlflow.convenience_functions import get_mlflow_results_artifacts, aggregate_fractional_splitter_results, aggregate_optimization_results
from util import parse_baseline_mutation_observations
from util import compute_delta_between_results
from visualization.plot_metric_for_dataset import barplot_metric_comparison, barplot_metric_comparison_bar_splitting, errorplot_metric_comparison, barplot_metric_comparison_bar
from visualization.plot_metric_for_dataset import barplot_metric_comparison_bar_splitting
from visualization.plot_metric_for_dataset import barplot_metric_functional_mutation_comparison
from visualization.plot_metric_for_dataset import barplot_metric_augmentation_comparison, barplot_metric_mutation_comparison
from visualization.plot_metric_for_dataset import barplot_metric_mutation_matrix
from visualization.plot_metric_for_dataset import scatterplot_metric_threshold_comparison
from visualization.plot_metric_for_dataset import threshold_metric_comparison
from typing import List


def plot_metric_comparison(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter,
                            dimension=None,
                            dim_reduction=None) -> None:
    results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, dim=dimension, dim_reduction=dim_reduction)
    cvtype = train_test_splitter.get_name() + f"d={dimension}_{dim_reduction}"   
    for metric in metrics:
        if metric == MSE:
            barplot_metric_comparison(metric_values=results_dict, cvtype=cvtype, metric=metric)
        if metric == SPEARMAN_RHO:
            errorplot_metric_comparison(metric_values=results_dict, cvtype=cvtype, metric=metric, plot_reference=True)


def load_cached_results(cached_results_filename: str) -> dict:
    with open(cached_results_filename, "rb") as infile:
        results_dict = pickle.load(infile)
    return results_dict


def load_results_dict_from_mlflow(datasets: List[str], 
                            algos: List[str],
                            metrics: List[str],  
                            reps: List[str],
                            train_test_splitter: list,
                            seeds: List[int]=None,
                            cache=False,
                            cache_fname=None,
                            optimized=True,
                            dim=None,
                            dim_reduction=LINEAR) -> dict:
    results_dict = {}
    fractional_splitter_results = []
    optimization_dict = None
    for splitter in train_test_splitter:
        if "optimization" in splitter.get_name().lower():
            optimization_dict = get_mlflow_results_optimization(datasets=datasets, algos=algos, reps=reps, metrics=metrics+[OBSERVED_Y, STD_Y], seeds=seeds)
        elif "fraction" in splitter.get_name().lower():
            _dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, augmentation=[None], dim=dim, dim_reduction=dim_reduction)
            fractional_splitter_results.append(_dict)
        else:
            _dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, optimized=optimized, augmentation=[None], dim=dim, dim_reduction=dim_reduction)
            results_dict[splitter.get_name()] = _dict
    if fractional_splitter_results:
        _fractional_results = aggregate_fractional_splitter_results(fractional_splitter_results, metrics=metrics)
        results_dict["Fractional"] = _fractional_results
    if optimization_dict:
        _opt_results = aggregate_optimization_results(optimization_dict, metrics=metrics)
        results_dict["Optimization"] = _opt_results
    if cache and cache_fname:
        with open(cache_fname, "wb") as outfile:
            pickle.dump(results_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return results_dict


def plot_metric_comparison_bar(datasets: List[str], 
                            algos: List[str],
                            metrics: List[str],  
                            reps: List[str],
                            train_test_splitter: list,
                            seeds: List[int]=None,
                            color_by="algo",
                            x_axis="algo",
                            cached_results=False,
                            optimized=True,
                            dim=None,
                            dim_reduction=LINEAR,
                            savefig=True) -> None:
    cached_filename = f"/Users/rcml/protein_regression/results/cache/results_comparison_d={'_'.join(datasets)}_a={'_'.join(algos)}_r={'_'.join(reps)}_m={'_'.join(metrics)}_s={'_'.join([s.get_name()[:5] for s in train_test_splitter[:5]])}_{str(seeds)}_opt={str(optimized)}_d={dim}_{dim_reduction}.pkl"
    if cached_results and exists(cached_filename):
        results_dict = load_cached_results(cached_filename)
    else:
        results_dict = load_results_dict_from_mlflow(datasets, algos, metrics, reps, train_test_splitter, seeds, cache=cached_results, cache_fname=cached_filename, optimized=optimized, dim=dim, dim_reduction=dim_reduction)
    cvtype = str(set([splitter.get_name()[:12] for splitter in train_test_splitter])) + f"d=full"   
    for metric in metrics:
        barplot_metric_comparison_bar(metric_values=results_dict, cvtype=cvtype, metric=metric, color_by=color_by, x_axis=x_axis, dim=dim, savefig=savefig)


def plot_metric_comparison_bar_param_delta(datasets: List[str], 
                            algos: List[str],
                            metrics: List[str],  
                            reps: List[str],
                            train_test_splitter: list,
                            seeds: List[int]=None,
                            color_by="algo",
                            x_axis="algo",
                            cached_results=False,
                            dim=None,
                            dim_reduction=LINEAR,
                            savefig=True) -> None:
    comparative_results_lst = []
    for opt_val in [True, False]:
        cached_filename = f"/Users/rcml/protein_regression/results/cache/results_comparison_d={'_'.join(datasets)}_a={'_'.join(algos)}_r={'_'.join(reps)}_m={'_'.join(metrics)}_s={'_'.join([s.get_name()[:5] for s in train_test_splitter[:5]])}_{str(seeds)}_opt={str(opt_val)}_d={dim}_{dim_reduction}.pkl"
        if cached_results and exists(cached_filename):
            results_dict = load_cached_results(cached_filename)
        else:
            results_dict = load_results_dict_from_mlflow(datasets, algos, metrics, reps, train_test_splitter, seeds, cache=cached_results, cache_fname=cached_filename, optimized=opt_val, dim=dim, dim_reduction=dim_reduction)
        comparative_results_lst.append(results_dict)
    results_dict = compute_delta_between_results(comparative_results_lst)
    cvtype = str(set([splitter.get_name()[:12] for splitter in train_test_splitter])) + f"d=full"   
    for metric in [SPEARMAN_RHO, "R2"]: # NOTE: we have already computed the delta, s.t. MSE is already 1-NMSE difference, hence MSE no longer required
        barplot_metric_comparison_bar(metric_values=results_dict, cvtype=cvtype, metric=metric, color_by=color_by, x_axis=x_axis, suffix="_opt_delta", savefig=savefig)


def plot_metric_comparison_bar_splitting(datasets: List[str], 
                            algos: List[str],
                            metrics: List[str],  
                            reps: List[str],
                            train_test_splitter: list,
                            seeds: List[int]=None,
                            color_by="algo",
                            x_axis="algo",
                            cached_results=False,
                            optimized=True,
                            savefig=True) -> None:
    cached_filename = f"/Users/rcml/protein_regression/results/cache/results_comparison_d={'_'.join(datasets)}_a={'_'.join(algos)}_r={'_'.join(reps)}_m={'_'.join(metrics)}_s={'_'.join([s.get_name()[:5] for s in train_test_splitter[:5]])}_{str(seeds)}_opt={str(optimized)}.pkl"
    if cached_results and exists(cached_filename):
        results_dict = load_cached_results(cached_filename)
    else:
        results_dict = load_results_dict_from_mlflow(datasets, algos, metrics, reps, train_test_splitter, seeds, cache=cached_results, cache_fname=cached_filename, optimized=optimized)
    cvtype = str(set([splitter.get_name()[:12] for splitter in train_test_splitter])) + f"d=full"   
    for metric in metrics:
        barplot_metric_comparison_bar_splitting(metric_values=results_dict, cvtype=cvtype, metric=metric, color_by=color_by, x_axis=x_axis, vline=False, legend=False, n_quantiles=4, savefig=savefig)


def load_mutation_results_with_baseline(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter: List[AbstractTrainTestSplitter],
                            dimension: int=None,
                            dim_reduction: str=None,
                            cached_results: bool=False) -> dict:
    cached_filename = f"/Users/rcml/protein_regression/results/cache/results_mutation_comparison_d={'_'.join(datasets)}_a={'_'.join(algos)}_r={'_'.join(reps)}_m={'_'.join(metrics)}_s={'_'.join([s.get_name()[:5] for s in train_test_splitter[:5]])}.pkl"
    if cached_results and exists(cached_filename):
        results_dict = load_cached_results(cached_filename)
    else: # TODO: refactor into dedicated loading function
        results_dict = {}
        for splitter in train_test_splitter:
            _dict = get_mlflow_results_artifacts(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, dim=dimension, dim_reduction=dim_reduction)
            added_train_observations, test_observations, train_trues = parse_baseline_mutation_observations(_dict)
            for method in _dict.get(datasets[0]).keys():
                _dict[datasets[0]][method]["additive"] = {None: {}}
                for split, (train_obs, test_obs, train_data) in enumerate(zip(added_train_observations, test_observations, train_trues)):
                    _dict[datasets[0]][method]["additive"][None][split] = {"trues": test_obs, "pred": train_obs, "train_trues": train_data}
            results_dict[splitter.get_name()] = _dict
        if cached_results:
            with open(cached_filename, "wb") as outfile:
                pickle.dump(results_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return results_dict


def plot_metric_mutation_comparison(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter: List[AbstractTrainTestSplitter],
                            dimension: int=None,
                            dim_reduction: str=None,
                            cached_results: bool=False,
                            t: int=None,
                            equality: str=None) -> None:
    results_dict = load_mutation_results_with_baseline(datasets, algos, metrics, reps, train_test_splitter, dimension,
                            dim_reduction, cached_results)
    for metric in metrics:
        if metric not in [MSE, SPEARMAN_RHO, "comparative_NMSE", "base_MSE", "mse"]:
            continue
        barplot_metric_mutation_comparison(results_dict, datasets=datasets, dim=dimension, metric=metric, t=t, equality=equality)


def plot_mutation_comparison_matrix(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter: List[AbstractTrainTestSplitter],
                            dimension: int=None,
                            dim_reduction: str=None,
                            cached_results: bool=False,
                            t: int=None,
                            equality: str=None,
                            savefig=True):
    results_dict = load_mutation_results_with_baseline(datasets, algos, metrics, reps, train_test_splitter, dimension,
                            dim_reduction, cached_results)
    for metric in metrics:
        if metric not in [MSE, SPEARMAN_RHO, "comparative_NMSE", "base_MSE", "mse"]:
            continue
        barplot_metric_mutation_matrix(results_dict, datasets=datasets, dim=dimension, metric=metric, savefig=savefig)
    


def plot_metric_functional_threshold_comparison(datasets: List[str],
                                    algos: List[str],
                                    metrics: List[str],
                                    reps: List[str],
                                    train_test_splitter: List[AbstractTrainTestSplitter]) -> None:
    """
    Load Benchmark datasets, each method performance on representations.
    Filter by functional thresholds, provide list with one value per dataset
    """
    results_dict = {}
    for splitter in train_test_splitter:
        _dict = get_mlflow_results_artifacts(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, dim=None, augmentation=[None], dim_reduction=LINEAR)
        results_dict[splitter.get_name()] = _dict
        # repurpose mutation comparison plot for vertical barplot comparison
    # scatterplot_metric_threshold_comparison(results_dict, dim=dimension, datasets=datasets, metric=metrics[0], thresholds=functional_thresholds)
    threshold_metric_comparison(results_dict, metric=metrics, datasets=datasets)


def plot_metric_augmentation_comparison(datasets: List[str], 
                                        algos: List[str],
                                        metrics: List[str], 
                                        reps: List[str],
                                        train_test_splitter: List[AbstractTrainTestSplitter], 
                                        augmentation: str, 
                                        dim=None, dim_reduction=None, reference_bars=True) -> None:
    results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, 
                                    augmentation=augmentation, dim=dim, dim_reduction=dim_reduction)
    ref_results_dict = {}
    if reference_bars:
        ref_results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, dim=dim, dim_reduction=dim_reduction)
    barplot_metric_augmentation_comparison(metric_values=results_dict, cvtype=train_test_splitter, augmentation=augmentation, metric=MSE, 
                                    dim=dim, dim_reduction=dim_reduction, reference_values=ref_results_dict)      


if __name__ == "__main__":
    algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
             RandomForest().get_name(), KNN().get_name()]
    metrics = [MSE, SPEARMAN_RHO] # MSE, SPEARMAN_RHO
    seeds = [11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245]
    dim = None
    dim_reduction = LINEAR # LINEAR, NON_LINEAR
    ### MAIN FIGURES
    # compare embeddings:
    plot_metric_comparison_bar(datasets=["1FQG",  "UBQT", "TIMB", "MTH3", "BRCA"], # ["1FQG",  "UBQT", "TIMB", "MTH3", "BRCA"]
                          reps=[ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM],
                          metrics=metrics,
                          train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")],
                          algos=[GPonRealSpace(kernel_factory=lambda: Matern52()).get_name()],
                          color_by="rep",
                          x_axis="rep",
                          cached_results=True)
    # # compare regressors:
    # plot_metric_comparison_bar(datasets=["1FQG",  "UBQT", "TIMB", "MTH3", "BRCA"],
    #                       reps=[ESM], metrics=metrics,
    #                       train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")],
    #                       algos=algos,
    #                       color_by="algo",
    #                       x_axis="algo",
    #                       cached_results=True)
    # # compare splitters:
    fractional_splitters = FractionalSplitterFactory("1FQG")
    plot_metric_comparison_bar_splitting(datasets=["1FQG", "UBQT"],
                          reps=[ESM],
                          metrics=metrics,
                          train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")] + fractional_splitters + [OptimizationSplitter("1FQG")],
                          algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()],
                          color_by="task",
                          x_axis="task",
                          seeds=seeds,
                          cached_results=True)
    # # # MUTATIONSPLITTER BENCHMARK PLOT
    # plot_mutation_comparison_matrix(datasets=["TOXI"], 
    #                     algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], metrics=[SPEARMAN_RHO, "base_MSE"], 
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 1, 1), BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 2, 3), BioSplitter("TOXI", 3, 3), BioSplitter("TOXI", 3, 4)],
    #                     dimension=None, dim_reduction=None,
    #                     cached_results=True)
    # ## MUTATIONSPLITTING SI: regular MSE
    # plot_mutation_comparison_matrix(datasets=["TOXI"], 
    #                     algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], 
    #                     metrics=[MSE], 
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 1, 1), BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 2, 3), BioSplitter("TOXI", 3, 3), BioSplitter("TOXI", 3, 4)],
    #                     dimension=None, 
    #                     dim_reduction=None,
    #                     cached_results=True)
    # MUTATIONSPLITTER BENCHMARK metrics on functional (<=0.5)
    # plot_metric_mutation_comparison(datasets=["TOXI"], algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], metrics=[MSE, SPEARMAN_RHO, "base_MSE", MEAN_Y], 
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 3)], # NOTE: there are no functional observations in 4M case, since there are not functional quad.variants
    #                     dimension=None, dim_reduction=None,
    #                     cached_results=True, t=-0.5, equality="<")
    # plot_metric_mutation_comparison(datasets=["TOXI"], algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], metrics=[MSE, SPEARMAN_RHO, "base_MSE", MEAN_Y], # "comparative_NMSE", 
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 3, 3)],
    #                     dimension=None, dim_reduction=None,
    #                     cached_results=True, t=-0.5, equality="<")
    # # metrics on non-functional (>= 0.5)
    # plot_metric_mutation_comparison(datasets=["TOXI"], algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], metrics=[MSE, SPEARMAN_RHO, "base_MSE", "comparative_NMSE", MEAN_Y], 
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 3), BioSplitter("TOXI", 3, 4)],
    #                     dimension=None, dim_reduction=None,
    #                     cached_results=True, t=-0.5, equality=">")
    # plot_metric_mutation_comparison(datasets=["TOXI"], algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], metrics=[MSE, SPEARMAN_RHO, "base_MSE", "comparative_NMSE", MEAN_Y], # "comparative_NMSE", 
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 3, 3)],
    #                     dimension=None, dim_reduction=None,
    #                     cached_results=True, t=-0.5, equality=">")
    # plot_mutation_comparison_matrix(datasets=["TOXI"], algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], metrics=[SPEARMAN_RHO, "base_MSE"],
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 3, 3)],
    #                     dimension=None, dim_reduction=None,
    #                     cached_results=True,)

    # ### SI: delta in performance:    # TODO
    # plot_metric_comparison_bar(datasets=["1FQG",  "UBQT", "TIMB", "MTH3", "BRCA"],
    #                       reps=[ESM], metrics=metrics,
    #                       train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")],
    #                       algos=algos,
    #                       color_by="algo",
    #                       x_axis="algo",
    #                       cached_results=True,
    #                       optimized=False) # TODO: debug w.r.t. Mlflow loading
    ### SUPPLEMENTARY FIGURES
    # # RANDOMSPLITTER BENCHMARK PLOT
    # plot_metric_comparison(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "BRCA", "TOXI"], 
    #                     algos=algos, metrics=metrics, reps=[ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM], 
    #                     train_test_splitter=RandomSplitter("1FQG"), dimension=None, dim_reduction=None)
    # # POSITIONSPLITTER BENCHMARK PLOT
    # plot_metric_comparison(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "BRCA"], 
    #                     algos=algos, metrics=metrics, reps=[ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM], 
    #                     train_test_splitter=PositionSplitter("1FQG"), dimension=None, dim_reduction=None)
    # # AUGMENTATION RANDOMSPLITTER # NOTE: EXCLUDED
    # plot_metric_augmentation_comparison(datasets=["1FQG", "UBQT",  "CALM"], algos=algos, metrics=[MSE],
    #                                     reps=[ONE_HOT, EVE, TRANSFORMER, ESM],
    #                                     dim=None, train_test_splitter=RandomSplitter("1FQG"),
    #                                     augmentation=[ROSETTA, EVE_DENSITY], reference_bars=True)
    # # AUGMENTATION POSITIONSPLITTER
    # plot_metric_augmentation_comparison(datasets=["1FQG", "UBQT", "CALM"], algos=algos, metrics=[MSE],
    #                                     reps=[ONE_HOT, EVE, TRANSFORMER, ESM],
    #                                     dim=None, train_test_splitter=PositionSplitter("1FQG"),
    #                                     augmentation=[ROSETTA, EVE_DENSITY], reference_bars=True)