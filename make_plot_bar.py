import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, REPRESENTATION, ROSETTA, TRANSFORMER, VAE, SPLIT, ONE_HOT, NONSENSE, KNN_name, VAE_DENSITY, NO_AUGMENT
from util.mlflow.convenience_functions import find_experiments_by_tags
from data.load_dataset import get_wildtype, get_alphabet
from visualization.plot_metric_for_dataset import barplot_metric_augmentation_comparison, barplot_metric_comparison
from typing import List

# gathers all our results and saves them into a numpy array
train_test_splitter = BlockPostionSplitter # RandomSplitter #
last_result_length = None

def plot_metric_comparison(datasets: List[str] = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"], algos=[GPonRealSpace().get_name(), GPonRealSpace(kernel=SquaredExponential()).get_name(), RandomForest().get_name(), KNN().get_name()],
                            metric=MSE, train_test_splitter=BlockPostionSplitter, reps=[ONE_HOT, VAE, TRANSFORMER]):
    results_dict = {}
    for dataset in datasets:
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}' and tags.{SPLIT} = '{train_test_splitter(dataset).get_name()}' and tags.{AUGMENTATION} = '{NO_AUGMENT}'"
                exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
                runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, 
                                        max_results=1, run_view_type=ViewType.ACTIVE_ONLY)
                assert len(runs) == 1 , rep+a+dataset
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


def plot_metric_augmentation_comparison(datasets: List[str]=["UBQT", "1FQG", "CALM"], reps = [ONE_HOT, TRANSFORMER],
                                        algos = [GPonRealSpace(kernel=SquaredExponential()).get_name(), RandomForest().get_name()], 
                                        metric=MSE, train_test_splitter=BlockPostionSplitter, augmentation = [ROSETTA, VAE_DENSITY]):
    results_dict = {}
    for dataset in datasets:
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                aug_results = {}
                for aug in augmentation:
                    filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}' and tags.{SPLIT} = '{train_test_splitter(dataset).get_name()}' and tags.{AUGMENTATION} = '{aug}'"
                    exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
                    runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, max_results=1, 
                                                run_view_type=ViewType.ACTIVE_ONLY)
                    assert len(runs) == 1 , rep+a+dataset+str(augmentation)
                    results = []      
                    for id in runs['run_id'].to_list():
                        for r in mlflow.tracking.MlflowClient().get_metric_history(id, metric):
                            results.append(r.value)
                    aug_results[aug] = results
                reps_results[rep] = aug_results
            if a == 'GPsquared_exponential':
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    barplot_metric_augmentation_comparison(metric_values=results_dict, cvtype=train_test_splitter(dataset).get_name(), augmentation=augmentation)      

if __name__ == "__main__":
    plot_metric_comparison()
    plot_metric_augmentation_comparison()
