import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential, Matern52
from mlflow.entities import ViewType
from algorithm_factories import UncertainRFFactory
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import Uncertain_RandomForestRegressor, UncertainRandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, LINEAR, METHOD, MSE, NON_LINEAR, REPRESENTATION, ROSETTA, TRANSFORMER, VAE, ESM
from util.mlflow.constants import SPLIT, ONE_HOT, NONSENSE, KNN_name, VAE_DENSITY, NO_AUGMENT
from util.mlflow.convenience_functions import find_experiments_by_tags, get_mlflow_results
from data.load_dataset import get_wildtype, get_alphabet
from visualization.plot_metric_for_dataset import barplot_metric_augmentation_comparison, barplot_metric_comparison
from typing import List

# gathers all our results and saves them into a numpy array
train_test_splitter =  BlockPostionSplitter # RandomSplitter #
last_result_length = None

def plot_metric_comparison(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            dimension=None,
                            train_test_splitter=train_test_splitter,
                            dim_reduction=None):
    results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, dim=dimension, dim_reduction=dim_reduction)
    # then calls #plot_metric_for_dataset
    cvtype = train_test_splitter(datasets[-1]).get_name() + f"d={dimension}_{dim_reduction}"
    barplot_metric_comparison(metric_values=results_dict, cvtype=cvtype, metric=MSE)

def plot_metric_augmentation_comparison(datasets: List[str]=["UBQT", "CALM", "1FQG"], 
                                        reps = [ONE_HOT, VAE, TRANSFORMER],
                                        algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), RandomForest().get_name()], # 
                                        metrics=[MSE], train_test_splitter=train_test_splitter, 
                                        augmentation = [ROSETTA, VAE_DENSITY]):
    results_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=train_test_splitter, augmentation=augmentation)
    barplot_metric_augmentation_comparison(metric_values=results_dict, cvtype=train_test_splitter(datasets[-1]).get_name(), augmentation=augmentation, metric=MSE)      

if __name__ == "__main__":
    datasets = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"]
    algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
             RandomForest().get_name(), KNN().get_name()]
    metrics = [MSE]
    representations = [ONE_HOT, VAE, TRANSFORMER, ESM]
    dim = None
    dim_reduction = LINEAR # LINEAR, NON_LINEAR
    # for dim in [10, 100, 1000]:
    #     plot_metric_comparison(datasets=datasets, algos=algos, metrics=metrics, reps=representations, dimension=dim, dim_reduction=dim_reduction)
    
    plot_metric_comparison(datasets=datasets, algos=algos, metrics=metrics, reps=representations, dimension=dim, dim_reduction=dim_reduction)
    # plot_metric_augmentation_comparison()
