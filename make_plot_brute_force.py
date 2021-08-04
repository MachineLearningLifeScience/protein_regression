import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from util.mlflow.constants import DATASET, METHOD, MSE, REPRESENTATION, TRANSFORMER, VAE, SPLIT
from util.mlflow.convenience_functions import find_experiments_by_tags
from visualization.plot_metric_for_dataset import plot_metric_for_dataset

# gathers all our results and saves them into a numpy array
datasets = ["CALM", "BRCA"]
train_test_splitter = BlockPostionSplitter #RandomSplitter # BlockPostionSplitter #RandomSplitter
metric = MSE
representations = [VAE, TRANSFORMER]
results_dict = {}
last_result_length = None

simons_algos = {VAE: [GPonRealSpace(), GPonRealSpace(kernel=SquaredExponential())],
                TRANSFORMER: [GPonRealSpace(), GPonRealSpace(kernel=SquaredExponential())]}

for dataset in datasets:
    result_dict = {}
    for repr in simons_algos.keys():
        for a in simons_algos[repr]:
            exps = find_experiments_by_tags({DATASET: dataset, 
                                             METHOD: a.get_name(), 
                                             REPRESENTATION: repr,
                                             SPLIT: train_test_splitter(dataset).get_name()})
            assert(len(exps) == 1)
            runs = mlflow.search_runs(experiment_ids=[exps[0].experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
            results = []
            for id in runs['run_id'].to_list():
                for r in mlflow.tracking.MlflowClient().get_metric_history(id, metric):
                    results.append(r.value)
            # if len(result_list) > 0:
            #     assert(len(results) == len(result_list[-1]))
            result_dict[repr+' '+a.get_name()] = results
    results_dict[dataset] = result_dict

print(results_dict)
# then calls #plot_metric_for_dataset
plot_metric_for_dataset(metric_values=results_dict, cvtype=train_test_splitter(dataset).get_name())


