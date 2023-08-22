from math import prod
import pytest
import pickle
from itertools import product
import numpy as np
from typing import List
from os.path import exists
from algorithms import KNN, GPonRealSpace, UncertainRandomForest, RandomForest
from gpflow.kernels import Matern52, SquaredExponential
from util.mlflow.constants import EVE, ESM, EVE_DENSITY, ONE_HOT, TRANSFORMER, LINEAR
from util.mlflow.constants import OBSERVED_Y, MSE, SPEARMAN_RHO
from util.mlflow.convenience_functions import load_results_dict_from_mlflow, get_mlflow_results_artifacts
from util.mlflow.convenience_functions import get_mlflow_results_optimization
from make_plot_bar import load_cached_results
from protocol_factories import BioSplitterFactory, RandomSplitterFactory, PositionalSplitterFactory, FractionalSplitterFactory
from data import BioSplitter, RandomSplitter, PositionSplitter
from algorithm_factories import KNNFactory, RandomForestFactory, GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory
from algorithm_factories import get_key_for_factory


method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory, GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory]]

test_datasets = ["1FQG", "UBQT", "TIMB", "MTH3", "BRCA"]
test_representations = [ONE_HOT, EVE, TRANSFORMER, ESM, EVE_DENSITY]
test_methods = [UncertainRandomForest().get_name(), # TODO
                GPonRealSpace().get_name(),
                GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(), 
                GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(),
                KNN().get_name(),
                ]
test_protocol_factories = [RandomSplitterFactory, PositionalSplitterFactory] # TODO
test_mutation_protocols = [BioSplitterFactory("TOXI", 1, 1),
                        BioSplitterFactory("TOXI", 1, 2),
                        BioSplitterFactory("TOXI", 2, 2),
                        BioSplitterFactory("TOXI", 2, 3),
                        BioSplitterFactory("TOXI", 3, 3),
                        BioSplitterFactory("TOXI", 3, 4),
                        BioSplitterFactory("TOXI", 4, 4),
                        ]
test_metric = [MSE]
test_dims = [None] # 2, 10, 100, 1000]
test_dims_lower = [2, 10, 100, 1000]
seeds = [11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245]

### FULL TEST
result_set_one = list(product(test_datasets, test_representations, [GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], test_protocol_factories, test_metric, test_dims))
result_set_two = list(product(test_datasets, [ESM], test_methods, test_protocol_factories, test_metric, test_dims))
result_set_tasks = list(product(test_datasets[:2], [ESM], [GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], [FractionalSplitterFactory], test_metric, test_dims))
result_set_opt = list(product(test_datasets[:2], test_representations[:-1], test_methods[:-1], seeds))
result_set_toxi = list(product(["TOXI"], [ESM], [GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], test_mutation_protocols, [SPEARMAN_RHO], test_dims))


# @pytest.mark.parametrize("dataset,representation,method,protocol_factory,metric,dim", list(result_set_one + result_set_two + result_set_tasks + result_set_toxi))
# def test_main_results_exist(dataset, representation, method, protocol_factory, metric, dim):
#     """
#     Tested data-sets cover Fig.2, 3, parts of 5, Fig.6
#     """
#     protocols = protocol_factory(dataset) if type(protocol_factory) != list else protocol_factory
#     for protocol in protocols:
#         cached_filename = f"/Users/rcml/protein_regression/results/cache/results_comparison_d={dataset}_a={method}_r={representation}_m={metric}_s={protocol.get_name()[:5]}_d={dim}.pkl"
#         if ("pos" in protocol.get_name().lower() and dataset == "TOXI") or ("fract" in protocol.get_name().lower() and dataset == "TOXI"):
#             continue
#         if exists(cached_filename):
#             with open(cached_filename, "rb") as infile:
#                 results = pickle.load(infile)
#         else:
#             try:
#                 results = load_results_dict_from_mlflow(datasets=[dataset], algos=[method], metrics=[metric], reps=[representation], train_test_splitter=[protocol], cache=True, cache_fname=cached_filename)
#             except AttributeError as _e:
#                 assert False
#         # account for naming ideosynch
#         protocol_name = "Fractional" if "fract" in protocol.get_name().lower() else protocol.get_name()
#         method = "GPsqexp" if "squared_exponential" in method else method
#         #print(f"{protocol.get_name()} ; {dataset} ; {method} ; {representation} ; {metric}")
#         metric_result = results.get(protocol_name).get(dataset).get(method).get(representation).get(None).get(metric)
#         assert metric_result is not None
#         assert len(np.array(metric_result).flatten()) > 1 # more than one result per split


# @pytest.mark.parametrize("dataset,representation,method,seed", list(result_set_opt))
# def test_optimization_results_exist(dataset, representation, method, seed):
#         cache_filename = f"/Users/rcml/protein_regression/results/cache/results_optimization_d={dataset}_a={method}_r={representation}_s={seed}.pkl"
#         # ALL ALGO RESULTS:
#         if exists(cache_filename):
#             with open(cache_filename, "rb") as infile:
#                 results = pickle.load(infile)
#         else:
#             results = get_mlflow_results_optimization(datasets=[dataset], algos=[method], reps=[representation], metrics=[OBSERVED_Y], seeds=[seed])
#             with open(cache_filename, "wb") as outfile:
#                 pickle.dump(results, outfile)
#         method = "GPsqexp" if "squared_exponential" in method else method
#         assert results[seed][dataset][method][representation][None][OBSERVED_Y] is not None
#         assert len(np.array(results[seed][dataset][method][representation][None][OBSERVED_Y]).flatten()) > 1


### TESTING ABLATION MATERIAL COMPLETENESS:
# NOTE: adhering to the form: product(test_datasets, test_representations, test_methods, test_protocol_factories, test_metric, test_dims, dim_reduction)
test_set_ablation_mutation = list(product(["TOXI"], 
                            [ONE_HOT, EVE, ESM], 
                            [GPonRealSpace(kernel_factory= lambda: Matern52()).get_name()], 
                            [BioSplitter("TOXI", 1, 1), BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 2, 3), BioSplitter("TOXI", 3, 3), BioSplitter("TOXI", 3, 4)], 
                            [MSE],
                            test_dims, 
                            [LINEAR],
                            ))
### SI: performance across dimensions:
test_set_ablation_dims = list(product(
    ["1FQG",  "UBQT", "TIMB", "MTH3", "BRCA"],
    [ONE_HOT, EVE, TRANSFORMER, ESM],
    [GPonRealSpace(kernel_factory=lambda: Matern52()).get_name()],
    [RandomSplitter("1FQG"), PositionSplitter("1FQG")],
    [MSE, SPEARMAN_RHO],
    [1000, 100, 10],
    [LINEAR],
    ))
### SI: remaining embedding & regressor comparisons
test_set_ablation_embeddings_regressors = list(product(
    ["1FQG",  "UBQT", "TIMB", "MTH3", "BRCA"],
    [ONE_HOT, EVE, EVE_DENSITY, TRANSFORMER, ESM],
    [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), RandomForest().get_name(), KNN().get_name()],
    [RandomSplitter("1FQG"), PositionSplitter("1FQG")],
    [MSE, SPEARMAN_RHO],
    [None],
    [LINEAR],
))

ablation_test_iterator = test_set_ablation_mutation #+ test_set_ablation_dims + test_set_ablation_embeddings_regressors

@pytest.mark.parametrize("dataset,representation,method,protocol,metric,dim,dim_reduction", list(ablation_test_iterator))
def test_ablation_results_exist(dataset, representation, method, protocol, metric, dim, dim_reduction):
    cached_filename = f"/Users/rcml/protein_regression/results/cache/results_comparison_d={dataset}_a={method}_r={representation}_m={metric}_s={protocol.get_name()[:5]}_d={dim}_{dim_reduction}.pkl"
    if exists(cached_filename):
        results_dict = load_cached_results(cached_filename)
    else:
        results_dict = load_results_dict_from_mlflow([dataset], [method], [metric], [representation], [protocol], seeds=None, cache=True, cache_fname=cached_filename, optimized=True, dim=dim, dim_reduction=dim_reduction) 
    assert len(results_dict[protocol][dataset][method][representation][None][metric]) > 1
