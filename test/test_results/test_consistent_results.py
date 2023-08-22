import pickle
import pytest
from itertools import product
import numpy as np
from typing import List
from os.path import exists
from algorithms import KNN, GPonRealSpace, UncertainRandomForest
from gpflow.kernels import Matern52, SquaredExponential
from util.mlflow.constants import EVE, ONE_HOT, SPEARMAN_RHO, TRANSFORMER, LINEAR
from util.mlflow.constants import OBSERVED_Y
from util.mlflow.convenience_functions import load_results_dict_from_mlflow, get_mlflow_results_artifacts
from make_plot_bar import load_cached_results
from data.train_test_split import RandomSplitter, PositionSplitter
from algorithm_factories import KNNFactory, RandomForestFactory, GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory
from algorithm_factories import get_key_for_factory


method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory, GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory]]


### FULL TEST
test_datasets = ["1FQG", "UBQT", "CALM", "TIMB", "BRCA", "MTH3"]
test_representations = [ONE_HOT, EVE, TRANSFORMER]
test_methods = [UncertainRandomForest().get_name(), 
                GPonRealSpace().get_name(),
                KNN().get_name(),
                GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(), 
                GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name()]
test_protocols = [RandomSplitter("1FQG"), PositionSplitter("1FQG")]
test_metric = [SPEARMAN_RHO]
test_dims = [None] # 2, 10, 100, 1000]
test_iterator = product(test_datasets, test_representations, test_methods, test_protocols, test_metric, test_dims)

# ### SMALL TEST
# test_datasets = ["1FQG"]
# test_representations = [ONE_HOT]
# test_methods = [
#                 GPonRealSpace().get_name(),
#                 ]
# test_protocols = [RandomSplitter("1FQG"), PositionSplitter("1FQG")]
# test_metric = [SPEARMAN_RHO]
# test_dims = [None]
# test_iterator = product(test_datasets, test_representations, test_methods, test_protocols, test_metric, test_dims)

@pytest.mark.parametrize("dataset,representation,method,protocol,metric,dim", 
        list(test_iterator))
def test_rho_for_constant_predictions(dataset: List[str], representation: List[str], method: List[str], protocol: List[object], metric: List[str], dim: List[int], missing_pass=True):
    # DONE: refactor loading functions
    # DONE: load mlflow results
    # TODO: test that if rho is NaN, then prediction values are constant
    optimized = True
    dim_reduction = LINEAR
    cached_filename = f"/Users/rcml/protein_regression/results/cache/results_comparison_d={'_'.join(dataset)}_a={'_'.join(method)}_r={'_'.join(representation)}_m={'_'.join(metric)}_s={protocol.get_name()[:5]}_opt={str(optimized)}_d={dim}_{dim_reduction}.pkl"
    if exists(cached_filename):
        results_dict = load_cached_results(cached_filename)
    else:
        try:
            results_dict = get_mlflow_results_artifacts(datasets=[dataset], reps=[representation], metrics=[metric], algos=[method], train_test_splitter=protocol,
                                                        dim=dim, dim_reduction=dim_reduction, optimize=optimized)
        except AssertionError as e:
            print(f"No entries found for {dataset,representation,method,protocol,metric,dim}...")
            return missing_pass
        # with open(cached_filename, "wb") as outfile:
        #     pickle.dump(results_dict, outfile)
    if method == GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name():
        method = "GPsqexp"
    metric_results = np.array([results_dict[dataset][method][representation][None][s][metric] for s in results_dict[dataset][method][representation][None].keys()])
    predictions = np.array([results_dict[dataset][method][representation][None][s]['pred'] for s in results_dict[dataset][method][representation][None].keys()])
    nan_predictions = np.isnan(metric_results)
    if np.isnan(metric_results).sum() == 0:
        assert True # if no rho=NaN we're consistent
    # if there as NaNs test if each split which contains a NaN is consistent:
    for idx in np.where(nan_predictions)[0]:
        #assert len(np.unique(predictions[idx])) <= 2 # two distinct values or all the same:
        np.testing.assert_allclose(predictions[idx], np.repeat(predictions[idx][0], len(predictions[idx])), rtol=0.0039)