import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential, Matern52
from mlflow.entities import ViewType
from algorithm_factories import UncertainRFFactory
from data.train_test_split import FractionalRandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import Uncertain_RandomForestRegressor, UncertainRandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, LINEAR, METHOD, MSE, SPEARMAN_RHO, MLL, PAC_BAYES_EPS
from util.mlflow.constants import NON_LINEAR, REPRESENTATION, ROSETTA, TRANSFORMER, VAE, ESM, VAE_AUX, VAE_RAND
from util.mlflow.constants import SPLIT, ONE_HOT, NONSENSE, KNN_name, VAE_DENSITY, VAE_AUX, NO_AUGMENT, LINEAR, NON_LINEAR
from util.mlflow.convenience_functions import find_experiments_by_tags, get_mlflow_results
from data.load_dataset import get_wildtype, get_alphabet
from visualization.plot_metric_for_dataset import cumulative_performance_plot
from typing import List

# gathers all our results and saves them into a numpy array
testing_fractions = np.concatenate([np.arange(0.001, .3, 0.01), np.arange(.3, .6, 0.03), np.arange(.6, 1.05, 0.05)])
train_test_splitters = [FractionalRandomSplitter(n) for n in testing_fractions]


def plot_cumulative_comparison(datasets: List[str], 
                            algos: List[str],
                            metrics: str,  
                            reps: List[str],
                            dimension=None,
                            train_test_splitters=train_test_splitters,
                            dim_reduction=None):
    results_dict = {}
    for frac, splitter in zip(testing_fractions, train_test_splitters):
        splitter_dict = get_mlflow_results(datasets=datasets, algos=algos, reps=reps, metrics=metrics, train_test_splitter=splitter, dim=dimension, dim_reduction=dim_reduction)
        results_dict[frac] = splitter_dict
    cumulative_performance_plot(results_dict)


if __name__ == "__main__":
    datasets = ["1FQG"] # ["TOXI"] # "MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"
    algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
             UncertainRandomForest().get_name()]
    metrics = [MLL, MSE, SPEARMAN_RHO, PAC_BAYES_EPS] # MSE
    representations = [ESM] # # SPECIAL CASE [UBQT, BLAT]: VAE+"_clusterval" 
    dim = None
    dim_reduction = LINEAR # LINEAR, NON_LINEAR
    
    plot_cumulative_comparison(datasets=datasets, algos=algos, metrics=metrics, reps=representations, dimension=dim, dim_reduction=dim_reduction)
