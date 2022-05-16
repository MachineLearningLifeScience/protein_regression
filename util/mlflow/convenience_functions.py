import mlflow
import os
import warnings
from os.path import join
import mlflow
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from util.mlflow.constants import DATASET, METHOD, REPRESENTATION, SPLIT, AUGMENTATION, NO_AUGMENT, DIMENSION, LINEAR, NON_LINEAR, VAE
from data.train_test_split import AbstractTrainTestSplitter

mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("results", "mlruns")))


def make_experiment_name_from_tags(tags: dict):
    return "".join([t + "_" + tags[t] + "__" for t in tags.keys()])


def find_experiments_by_tags(tags: dict):
    exps = mlflow.tracking.MlflowClient().list_experiments()
    print(exps)
    def all_tags_match(e):
        for tag in tags.keys():
            if tag not in e.tags:
                return False
            if e.tags[tag] != tags[tag]:
                return False
        return True
    return [e for e in exps if all_tags_match(e)]


def get_mlflow_results(datasets: list, algos: list, reps: list, metrics: list, train_test_splitter: AbstractTrainTestSplitter, augmentation: list=[None], dim=None, dim_reduction=NON_LINEAR) -> dict:
    results_dict = {}
    for dataset in datasets:
        algo_results = {}
        for a in algos:
            reps_results = {}
            for rep in reps:
                aug_results = {}
                for aug in augmentation:
                    filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}' and tags.{SPLIT} = '{train_test_splitter(dataset).get_name()}'"
                    if aug:
                        filter_string += f" and tags.{AUGMENTATION} = '{aug}'"
                    if dim and not (rep==VAE and dim >= 30):
                        filter_string += f" and tags.{DIMENSION} = '{dim}' and tags.DIM_REDUCTION = '{dim_reduction}'"
                    exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
                    runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, max_results=1, run_view_type=ViewType.ACTIVE_ONLY)
                    assert len(runs) == 1 , rep+a+dataset+str(augmentation)
                    metric_results = {metric: [] for metric in metrics}     
                    for id in runs['run_id'].to_list():
                        for metric in metrics:
                            try:
                                for r in mlflow.tracking.MlflowClient().get_metric_history(id, metric):
                                    metric_results[metric].append(r.value)
                            except MlflowException as e:
                                continue
                    aug_results[aug] = metric_results
                reps_results[rep] = aug_results
            if a == 'GPsquared_exponential':
                a = "GPsqexp"
            algo_results[a] = reps_results
        results_dict[dataset] = algo_results
    return results_dict
