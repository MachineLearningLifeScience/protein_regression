import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, REPRESENTATION, ROSETTA, TRANSFORMER, VAE, SPLIT, ONE_HOT, NONSENSE, KNN_name
from util.mlflow.convenience_functions import find_experiments_by_tags
from data.load_dataset import get_wildtype, get_alphabet
from visualization.plot_metric_for_dataset import barplot_metric_comparison

# gathers all our results and saves them into a numpy array
#datasets = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"]
datasets = ["UBQT", "1FQG", "CALM"]
train_test_splitter = BlockPostionSplitter # RandomSplitter #  
metric = MSE
representations = [VAE, TRANSFORMER, ONE_HOT]
last_result_length = None
#reps = [ONE_HOT, VAE, TRANSFORMER]
reps = [ONE_HOT, TRANSFORMER]
augmentation = ROSETTA
#algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel=SquaredExponential()).get_name(), RandomForest().get_name(), KNN().get_name()]
algos = [GPonRealSpace(kernel=SquaredExponential()).get_name()]

results_dict = {}
for dataset in datasets:
    algo_results = {}
    for a in algos:
        reps_results = {}
        for rep in reps:
            filter_string = f"tags.{DATASET} = '{dataset}' and tags.{METHOD} = '{a}' and tags.{REPRESENTATION} = '{rep}' and tags.{SPLIT} = '{train_test_splitter(dataset).get_name()}' and tags.{AUGMENTATION} = '{augmentation}'"
            exps =  mlflow.tracking.MlflowClient().get_experiment_by_name(dataset)
            runs = mlflow.search_runs(experiment_ids=[exps.experiment_id], filter_string=filter_string, run_view_type=ViewType.ACTIVE_ONLY)
            assert len(runs) >= 1 , rep+a+dataset+str(augmentation)
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
barplot_metric_comparison(metric_values=results_dict, cvtype=train_test_splitter(dataset).get_name(), 
                        augmentation=augmentation)
