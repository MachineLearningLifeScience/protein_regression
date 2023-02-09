import mlflow
import numpy as np
from scipy.special import comb
from gpflow.kernels import SquaredExponential, Matern52
from mlflow.entities import ViewType
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from algorithm_factories import UncertainRFFactory
from data.train_test_split import PositionSplitter, RandomSplitter, BioSplitter, AbstractTrainTestSplitter, WeightedTaskSplitter, FractionalRandomSplitter, OptimizationSplitter
from protocol_factories import FractionalSplitterFactory
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import Uncertain_RandomForestRegressor, UncertainRandomForest
from algorithms.KNN import KNN
from algorithms.gmm_regression import GMMRegression
from util.mlflow.constants import AUGMENTATION, DATASET, LINEAR, METHOD, MSE, SPEARMAN_RHO
from util.mlflow.constants import NON_LINEAR, REPRESENTATION, ROSETTA, TRANSFORMER, VAE, ESM, VAE_AUX, VAE_RAND, EVE, EVE_DENSITY
from util.mlflow.constants import SPLIT, ONE_HOT, NONSENSE, KNN_name, VAE_DENSITY, VAE_AUX, NO_AUGMENT, LINEAR, NON_LINEAR, MEAN_Y, STD_Y, OBSERVED_Y
from util.mlflow.convenience_functions import find_experiments_by_tags, get_mlflow_results, get_mlflow_results_optimization
from util.mlflow.convenience_functions import get_mlflow_results_artifacts, aggregate_fractional_splitter_results, aggregate_optimization_results
from util import parse_baseline_mutation_observations
from visualization.plot_metric_for_dataset import barplot_metric_comparison, errorplot_metric_comparison, barplot_metric_comparison_bar
from visualization.plot_metric_for_dataset import barplot_metric_functional_mutation_comparison
from visualization.plot_metric_for_dataset import barplot_metric_augmentation_comparison, barplot_metric_mutation_comparison
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

def plot_metric_comparison_bar(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter: list,
                            seeds: List[int]=None,
                            color_by="algo",
                            x_axis="algo") -> None:
    results_dict = {}
    fractional_splitter_results = []
    optimization_dict = None
    for splitter in train_test_splitter:
        if "optimization" in splitter.get_name().lower():
            optimization_dict = get_mlflow_results_optimization(datasets=datasets, algos=algos, reps=reps, metrics=metrics+[OBSERVED_Y, STD_Y], seeds=seeds)
        elif "fraction" in splitter.get_name().lower():
            _dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, dim=None, augmentation=[None], dim_reduction=LINEAR)
            fractional_splitter_results.append(_dict)
        else:
            _dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, dim=None, augmentation=[None], dim_reduction=LINEAR)
            results_dict[splitter.get_name()] = _dict
    if fractional_splitter_results:
        _fractional_results = aggregate_fractional_splitter_results(fractional_splitter_results, metrics=metrics)
        results_dict["Fractional"] = _fractional_results
    if optimization_dict:
        _opt_results = aggregate_optimization_results(optimization_dict, metrics=metrics)
        results_dict["Optimization"] = _opt_results
    cvtype = str([splitter.get_name() for splitter in train_test_splitter]) + f"d=full"   
    for metric in metrics:
        barplot_metric_comparison_bar(metric_values=results_dict, cvtype=cvtype, metric=metric, color_by=color_by, x_axis=x_axis) # TODO


def plot_metric_mutation_comparison(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter: List[AbstractTrainTestSplitter],
                            dimension=None,
                            dim_reduction=None) -> None:
    results_dict = {}
    for splitter in train_test_splitter:
        _dict = get_mlflow_results_artifacts(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, dim=dimension, dim_reduction=dim_reduction)
        added_train_observations, test_observations = parse_baseline_mutation_observations(_dict)
        for method in _dict.get(datasets[0]).keys():
            _dict[datasets[0]][method]["additive"] = {None: {}}
            for split, (test_obs, train_obs) in enumerate(zip(added_train_observations, test_observations)):
                _dict[datasets[0]][method]["additive"][None][split] = {"trues": test_obs, "pred": train_obs}
        results_dict[splitter.get_name()] = _dict
    n_mutation_sites = 4
    n_combinations = np.cumsum([np.power(np.power(20, comb(n_mutation_sites, k)), k) for k in range(1, 5)]) # M = 20**(4 choose i)**i AND each domain sum of previous combinations (cumsum)
    barplot_metric_mutation_comparison(results_dict, dim=dimension, metric=SPEARMAN_RHO, N_combinations=n_combinations)


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
    # # compare embeddings:
    # plot_metric_comparison_bar(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "BRCA"],
    #                       reps=[ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM],
    #                       metrics=metrics,
    #                       train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")],
    #                       algos=[GPonRealSpace(kernel_factory=lambda: Matern52()).get_name()],
    #                       color_by="rep",
    #                       x_axis="rep")
    # # compare regressors:
    # plot_metric_comparison_bar(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "BRCA"],
    #                       reps=[ESM], metrics=metrics,
    #                       train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")],
    #                       algos=algos,
    #                       color_by="algo",
    #                       x_axis="algo")
    # compare splitters:
    fractional_splitters = FractionalSplitterFactory("1FQG")
    plot_metric_comparison_bar(datasets=["1FQG", "UBQT"],
                          reps=[ESM],
                          metrics=metrics,
                          train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")] + fractional_splitters + [OptimizationSplitter("1FQG")],
                          algos=[GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()],
                          color_by="task",
                          x_axis="task",
                          seeds=seeds)
    # MUTATIONSPLITTER BENCHMARK PLOT
    # plot_metric_mutation_comparison(datasets=["TOXI"], algos=[RandomForest().get_name(), GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(), KNN().get_name()], metrics=[MSE, SPEARMAN_RHO], 
    #                     reps=[ONE_HOT, EVE, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 2, 3), BioSplitter("TOXI", 3, 3)],
    #                     dimension=None, dim_reduction=None)

    ### SUPPLEMENTARY FIGURES
    # # RANDOMSPLITTER BENCHMARK PLOT
    # plot_metric_comparison(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "BRCA", "TOXI"], 
    #                     algos=algos, metrics=metrics, reps=[ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM], 
    #                     train_test_splitter=RandomSplitter("1FQG"), dimension=None, dim_reduction=None)
    # # POSITIONSPLITTER BENCHMARK PLOT
    # plot_metric_comparison(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "BRCA"], 
    #                     algos=algos, metrics=metrics, reps=[ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM], 
    #                     train_test_splitter=PositionSplitter("1FQG"), dimension=None, dim_reduction=None)
    # BENCHMARK GIVEN FUNCTIONAL THRESHOLD # TODO
    # for data in ["UBQT", "TOXI"]:
    #     for metric in [MSE, SPEARMAN_RHO]:
    #         plot_metric_functional_threshold_comparison(datasets=[data], # ["TOXI"]
    #                     algos=[RandomForest().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), KNN().get_name()], 
    #                     metrics=[metric], 
    #                     reps=[ONE_HOT, EVE, ESM, TRANSFORMER],
    #                     train_test_splitter=[RandomSplitter("1FQG")])
    # # No threshold for comparison:
    # plot_metric_functional_comparison(datasets=["TOXI"], 
    #                 algos=[RandomForest().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), KNN().get_name()], 
    #                 metrics=[SPEARMAN_RHO, MSE], 
    #                 reps=[ONE_HOT, EVE, ESM, TRANSFORMER],
    #                 train_test_splitter=[RandomSplitter("1FQG")],
    #                 dimension=None, dim_reduction=None)
    # # AUGMENTATION RANDOMSPLITTER
    # plot_metric_augmentation_comparison(datasets=["1FQG", "UBQT",  "CALM"], algos=algos, metrics=[MSE],
    #                                     reps=[ONE_HOT, EVE, TRANSFORMER, ESM],
    #                                     dim=None, train_test_splitter=RandomSplitter("1FQG"),
    #                                     augmentation=[ROSETTA, EVE_DENSITY], reference_bars=True)
    # # AUGMENTATION POSITIONSPLITTER
    # plot_metric_augmentation_comparison(datasets=["1FQG", "UBQT", "CALM"], algos=algos, metrics=[MSE],
    #                                     reps=[ONE_HOT, EVE, TRANSFORMER, ESM],
    #                                     dim=None, train_test_splitter=PositionSplitter("1FQG"),
    #                                     augmentation=[ROSETTA, EVE_DENSITY], reference_bars=True)