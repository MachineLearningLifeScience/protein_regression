import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from util.mlflow.constants import DATASET, METHOD, MSE, REPRESENTATION
from util.mlflow.convenience_functions import find_experiments_by_tags
from visualization.plot_metric_for_dataset import plot_metric_for_dataset

# gathers all our results and saves them into a numpy array
dataset = "1FQG"
metric = MSE
representations = [None, "transformer"]
algorithm_name_list = []
result_list = []
last_result_length = None


# Jacob's code

# Simon's code
simons_algos = {#None: [GPOneHotSequenceSpace(alphabet_size=0), GPOneHotSequenceSpace(alphabet_size=0, kernel=SquaredExponential())],
                "transformer": [GPonRealSpace(), GPonRealSpace(kernel=SquaredExponential())]}
for repr in simons_algos.keys():
    for a in simons_algos[repr]:
        exps = find_experiments_by_tags({DATASET: dataset, METHOD: a.get_name(), REPRESENTATION: repr})
        assert(len(exps) == 1)
        algorithm_name_list.append(a.get_name())
        runs = mlflow.search_runs(experiment_ids=[exps[0].experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
        results = []
        for id in runs['run_id'].to_list():
            for r in mlflow.tracking.MlflowClient().get_metric_history(id, metric):
                results.append(r.value)
        if len(result_list) > 0:
            assert(len(results) == len(result_list[-1]))
        result_list.append(results)


# Richard's code

# then calls #plot_metric_for_dataset
results = np.array(result_list)
plot_metric_for_dataset(dataset=dataset, metric=metric, metric_values=results, algorithm_names=algorithm_name_list)
