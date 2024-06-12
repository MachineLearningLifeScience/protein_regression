from gpflow.kernels import Matern52, SquaredExponential

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.KNN import KNN
from algorithms.uncertain_rf import UncertainRandomForest
from data.train_test_split import BioSplitter, PositionSplitter, RandomSplitter
from util.mlflow.constants import (ESM, ESM1V, ESM2, EVE, LINEAR, MSE,
                                   NO_AUGMENT, ONE_HOT, PROTT5, TRANSFORMER)
from visualization.plot_metric_for_uncertainties import plot_uncertainty_eval

datasets = ["1FQG"] # 1FQG, UBQT, CALM

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
dim_reduction = LINEAR
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