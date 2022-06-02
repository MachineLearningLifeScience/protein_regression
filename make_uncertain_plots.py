from os.path import join, dirname
import json
import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential, Matern52
from gpflow.kernels.linears import Linear
from data.train_test_split import BlockPostionSplitter, RandomSplitter, BioSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, REPRESENTATION, LINEAR, NON_LINEAR
from util.mlflow.constants import NO_AUGMENT, ROSETTA, TRANSFORMER, VAE, SPLIT, ONE_HOT, ESM, NONSENSE, KNN_name, VAE_DENSITY
from visualization.plot_metric_for_uncertainties import plot_uncertainty_eval

datasets = ["TOXI"]
train_test_splitter = BioSplitter("TOXI", inverse=True) # RandomSplitter # BlockPostionSplitter # BioSplitter
metric = MSE
reps = [TRANSFORMER, ESM, VAE, ONE_HOT]
augmentations =  [NO_AUGMENT]
number_quantiles = 10
algos = [GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(), 
        GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), UncertainRandomForest().get_name(), GPonRealSpace().get_name()]
d = None # 2, 10, 100, 1000, None
dim_reduction = LINEAR # LINEAR # NON_LINEAR


if __name__ == "__main__":
    plot_uncertainty_eval(datasets=datasets, reps=reps,
                          algos=algos, train_test_splitter=train_test_splitter, 
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True, 
                          d=d, dim_reduction=dim_reduction)