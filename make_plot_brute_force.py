import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from util.mlflow.constants import DATASET, METHOD, MSE, REPRESENTATION, TRANSFORMER, VAE, SPLIT, ONE_HOT, NONSENSE
from util.mlflow.convenience_functions import find_experiments_by_tags
from data.load_dataset import get_wildtype, get_alphabet
from visualization.plot_metric_for_dataset import plot_metric_for_dataset

# gathers all our results and saves them into a numpy array
datasets = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"]
#datasets = ["1FQG"]
train_test_splitter = BlockPostionSplitter #RandomSplitter
metric = MSE
representations = [VAE, TRANSFORMER, ONE_HOT, NONSENSE]
#representations = [ONE_HOT, NONSENSE]
results_dict = {}
last_result_length = None

algos = {VAE: [GPonRealSpace(), GPonRealSpace(kernel=SquaredExponential()), RandomForest()],
        TRANSFORMER: [GPonRealSpace(), GPonRealSpace(kernel=SquaredExponential()), RandomForest()],
        ONE_HOT: [GPOneHotSequenceSpace(alphabet_size=len(get_alphabet('BRCA'))), GPOneHotSequenceSpace(alphabet_size=len(get_alphabet('BRCA')), kernel=SquaredExponential()), RandomForest()],
        NONSENSE: [GPonRealSpace(), GPonRealSpace(kernel=SquaredExponential()), RandomForest()]}

for dataset in datasets:
    result_dict = {}
    for repr in algos.keys():
        for a in algos[repr]:
            exps = find_experiments_by_tags({DATASET: dataset, 
                                             METHOD: a.get_name(), 
                                             REPRESENTATION: repr,
                                             SPLIT: train_test_splitter(dataset).get_name()})
            assert len(exps) == 1, repr
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
