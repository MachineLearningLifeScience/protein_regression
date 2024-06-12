from os.path import join, dirname
import json
import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential, Matern52
from gpflow.kernels.linears import Linear
from sklearn.model_selection import train_test_split
from data.train_test_split import BlockPostionSplitter, PositionSplitter, RandomSplitter, BioSplitter, FractionalRandomSplitter, WeightedTaskSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, PROTT5, PSSM, REPRESENTATION, LINEAR, NON_LINEAR
from util.mlflow.constants import NO_AUGMENT, ROSETTA, TRANSFORMER, VAE, VAE_RAND, VAE_AUX, EVE, ESM2, ESM1V
from util.mlflow.constants import SPLIT, ONE_HOT, ESM, NONSENSE, KNN_name, VAE_DENSITY
from visualization.plot_metric_for_uncertainties import plot_uncertainty_eval, plot_uncertainty_eval_across_dimensions

datasets = ["1FQG"] # 1FQG, UBQT, CALM
# datasets = ["TOXI"]
# train_test_splitter = BioSplitter(datasets[0], 3, 4) # BlockPostionSplitter # PositionSplitter # RandomSplitter # BlockPostionSplitter # BioSplitter

metric = MSE
reps = [PROTT5, ESM, EVE, ONE_HOT]
augmentations =  [NO_AUGMENT]
number_quantiles = 10
algos = [UncertainRandomForest().get_name(), 
        GPonRealSpace().get_name(),
        KNN().get_name(),
        GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(), 
        GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(),]
d = None # 2, 10, 100, 1000, None
dim_reduction = LINEAR # LINEAR # NON_LINEAR
cached_results = True


if __name__ == "__main__":
        ### MAKE UNCERTAIN PLOTS RANDOMSPLITTER
        plot_uncertainty_eval(datasets=datasets, reps=reps,
                          algos=algos, train_test_splitter=RandomSplitter(datasets[0]),
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True,
                          d=d, dim_reduction=None, cached_results=cached_results)
        ### MAKE UNCERTAIN PLOTS POSITIONSPLITTER
        plot_uncertainty_eval(datasets=datasets, reps=reps,
                          algos=algos, train_test_splitter=PositionSplitter(datasets[0]), 
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True, 
                          d=d, dim_reduction=None, cached_results=cached_results)
        #########
        #### SUPPLEMENTARY:
        for dataset in ["1FQG", "UBQT", "TIMB", "MTH3", "BRCA"]:
                plot_uncertainty_eval(datasets=[dataset], 
                        reps=[TRANSFORMER, PROTT5, ESM, ESM1V, ESM2, EVE, ONE_HOT],
                        algos=algos, train_test_splitter=RandomSplitter(datasets[0]),
                        augmentations = augmentations, number_quantiles=number_quantiles, optimize=True,
                        d=d, dim_reduction=None, cached_results=cached_results)
        # MAKE UNCERTAIN PLOTS POSITIONSPLITTER
        for dataset in ["1FQG", "UBQT", "TIMB", "MTH3", "BRCA"]:
                plot_uncertainty_eval(datasets=[dataset], 
                        reps=[TRANSFORMER, PROTT5, ESM, ESM1V, ESM2, EVE, ONE_HOT],
                        algos=algos, train_test_splitter=PositionSplitter(datasets[0]), 
                        augmentations = augmentations, number_quantiles=number_quantiles, optimize=True, 
                        d=d, dim_reduction=None, cached_results=cached_results)
        ### SI: UNCERTAINTIES TOXI
        plot_uncertainty_eval(datasets=["TOXI"], reps=reps,
                          algos=algos, train_test_splitter=BioSplitter("TOXI", 1, 2),
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True,
                          d=d, dim_reduction=None, cached_results=cached_results)
        plot_uncertainty_eval(datasets=["TOXI"], reps=reps,
                          algos=algos, train_test_splitter=BioSplitter("TOXI", 2, 2),
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True,
                          d=d, dim_reduction=None, cached_results=cached_results)
        plot_uncertainty_eval(datasets=["TOXI"], reps=reps,
                          algos=algos, train_test_splitter=BioSplitter("TOXI", 2, 3),
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True,
                          d=d, dim_reduction=None, cached_results=cached_results)
        plot_uncertainty_eval(datasets=["TOXI"], reps=reps,
                          algos=algos, train_test_splitter=BioSplitter("TOXI", 3, 3),
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True,
                          d=d, dim_reduction=None, cached_results=cached_results)
        plot_uncertainty_eval(datasets=["TOXI"], reps=reps,
                          algos=algos, train_test_splitter=BioSplitter("TOXI", 3, 4),
                          augmentations = augmentations, number_quantiles=number_quantiles, optimize=True,
                          d=d, dim_reduction=None, cached_results=cached_results)
        # ### MAKE UNCERTAIN PLOTS ACROSS DIMENSIONS NOTE: taken out
        # plot_uncertainty_eval_across_dimensions(datasets=datasets, reps=reps,
        #                   algos=algos, train_test_splitter=RandomSplitter(datasets[0]),
        #                   augmentation = augmentations, number_quantiles=number_quantiles, optimize=True,
        #                   dimensions=[2, 10, 100, 1000, None], dim_reduction=dim_reduction)
        # plot_uncertainty_eval_across_dimensions(datasets=datasets, reps=reps,
        #                   algos=algos, train_test_splitter=PositionSplitter(datasets[0]),
        #                   augmentation = augmentations, number_quantiles=number_quantiles, optimize=True,
        #                   dimensions=[2, 10, 100, 1000, None], dim_reduction=dim_reduction)