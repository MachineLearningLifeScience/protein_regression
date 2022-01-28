import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import DATASET, METHOD, MSE, REPRESENTATION, TRANSFORMER, VAE, SPLIT, ONE_HOT, NONSENSE, KNN_name
from util.mlflow.convenience_functions import find_experiments_by_tags
from data.load_dataset import get_wildtype, get_alphabet
from visualization.plot_metric_for_dataset import barplot_metric_comparison

# gathers all our results and saves them into a numpy array
datasets = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"]
train_test_splitter = RandomSplitter # BlockPostionSplitter 
metric = MSE
representations = [VAE, TRANSFORMER, ONE_HOT]
last_result_length = None
reps = [ONE_HOT, VAE, TRANSFORMER]
algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel=SquaredExponential()).get_name(), RandomForest().get_name(), KNN().get_name()]

results_dict = {}
for dataset in datasets:
    algo_results = {}
    for a in algos:
        reps_results = {}
        for rep in reps:
            exps = find_experiments_by_tags({DATASET: dataset, 
                                             METHOD: a, 
                                             REPRESENTATION: rep,
                                             SPLIT: train_test_splitter(dataset).get_name()})
            assert len(exps) == 1, rep+a+dataset
            runs = mlflow.search_runs(experiment_ids=[exps[0].experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
            results = []
            for id in runs['run_id'].to_list():
                for r in mlflow.tracking.MlflowClient().get_metric_history(id, metric):
                    results.append(r.value)
            reps_results[rep] = results

        if a == 'GPsquared_exponential':
            a = "GPsqexp"
        algo_results[a] = reps_results
    results_dict[dataset] = algo_results
# then calls #plot_metric_for_dataset
barplot_metric_comparison(metric_values=results_dict, cvtype=train_test_splitter(dataset).get_name())
