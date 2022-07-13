import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential, Matern52
from mlflow.entities import ViewType
from algorithm_factories import UncertainRFFactory
from data.train_test_split import BlockPostionSplitter, PositionSplitter, RandomSplitter, BioSplitter, AbstractTrainTestSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import Uncertain_RandomForestRegressor, UncertainRandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, LINEAR, METHOD, MSE, SPEARMAN_RHO
from util.mlflow.constants import NON_LINEAR, REPRESENTATION, ROSETTA, TRANSFORMER, VAE, ESM, VAE_AUX, VAE_RAND, EVE, EVE_DENSITY
from util.mlflow.constants import SPLIT, ONE_HOT, NONSENSE, KNN_name, VAE_DENSITY, VAE_AUX, NO_AUGMENT, LINEAR, NON_LINEAR
from util.mlflow.convenience_functions import find_experiments_by_tags, get_mlflow_results
from visualization.plot_metric_for_dataset import barplot_metric_comparison, errorplot_metric_comparison
from visualization.plot_metric_for_dataset import barplot_metric_augmentation_comparison, barplot_metric_mutation_comparison
from typing import List


def plot_metric_comparison(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter,
                            dimension=None,
                            dim_reduction=None):
    results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, dim=dimension, dim_reduction=dim_reduction)
    # then calls #plot_metric_for_dataset
    cvtype = train_test_splitter.get_name() + f"d={dimension}_{dim_reduction}"
    # if metric not in [MSE, SPEARMAN_RHO]:
    #     raise NotImplementedError("No plotting for this metric exists!")
    for metric in metrics:
        if metric == MSE:
            barplot_metric_comparison(metric_values=results_dict, cvtype=cvtype, metric=metric)
        if metric == SPEARMAN_RHO:
            errorplot_metric_comparison(metric_values=results_dict, cvtype=cvtype, metric=metric, plot_reference=False)


def plot_metric_mutation_comparison(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            train_test_splitter: List[AbstractTrainTestSplitter],
                            dimension=None,
                            dim_reduction=None):
    results_dict = {}
    for splitter in train_test_splitter:
        _dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, dim=dimension, dim_reduction=dim_reduction)
        results_dict[splitter.get_name()] = _dict
    barplot_metric_mutation_comparison(results_dict)


# def plot_metric_augmentation_comparison(datasets: List[str]=["UBQT", "CALM", "1FQG"], 
#                                         reps = [ONE_HOT, VAE, TRANSFORMER, ESM],
#                                         algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), RandomForest().get_name()], # 
#                                         metrics=[MSE], train_test_splitter=train_test_splitter, 
#                                         augmentation = [ROSETTA, VAE_DENSITY], 
#                                         dim=None, dim_reduction=LINEAR, reference_bars=True):
#     results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, 
#                                     augmentation=augmentation, dim=dim, dim_reduction=dim_reduction)
#     if reference_bars:
#         ref_results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, dim=dim, dim_reduction=dim_reduction)
#     barplot_metric_augmentation_comparison(metric_values=results_dict, cvtype=train_test_splitter(datasets[-1]).get_name(), augmentation=augmentation, metric=MSE, 
#                                     dim=dim, dim_reduction=dim_reduction, reference_values=ref_results_dict)      


if __name__ == "__main__":
    algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
             RandomForest().get_name(), KNN().get_name()]
    metrics = [MSE, SPEARMAN_RHO] # MSE
    dim = None
    dim_reduction = LINEAR # LINEAR, NON_LINEAR
    # RANDOMSPLITTER BENCHMARK PLOT
    plot_metric_comparison(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA", "TOXI"], 
                        algos=algos, metrics=metrics, reps=[ONE_HOT, EVE, EVE_DENSITY, VAE_AUX, TRANSFORMER, ESM], 
                        train_test_splitter=RandomSplitter("1FQG"), dimension=None, dim_reduction=LINEAR)
    # # POSITIONSPLITTER BENCHMARK PLOT
    # plot_metric_comparison(datasets=["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"], 
    #                     algos=algos, metrics=metrics, reps=[ONE_HOT, EVE, EVE_DENSITY, VAE_AUX, TRANSFORMER, ESM], 
    #                     train_test_splitter=PositionSplitter("1FQG"), dimension=None, dim_reduction=LINEAR)
    # # MUTATIONSPLITTER BENCHMARK PLOT
    # plot_metric_mutation_comparison(datasets=["TOXI"], algos=algos, metrics=[MSE], 
    #                     reps=[ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM],
    #                     train_test_splitter=[BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 2, 3), BioSplitter("TOXI", 3, 3), BioSplitter("TOXI", 3, 4)],
    #                     dimension=None, dim_reduction=LINEAR)
    # TODO: below
    # plot_metric_augmentation_comparison(dim=None)
