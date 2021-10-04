import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential
from mlflow.entities import ViewType
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import DATASET, METHOD, MSE, REPRESENTATION, TRANSFORMER, VAE, SPLIT, ONE_HOT, NONSENSE, KNN_name, SEED, OPTIMIZATION,\
     EXPERIMENT_TYPE, OBSERVED_Y
from util.mlflow.convenience_functions import find_experiments_by_tags
from data.load_dataset import load_dataset
from visualization.plot_metric_for_dataset import plot_optimization_task

# gathers all our results and saves them into a numpy array
datasets = ["1FQG"]
representations = [TRANSFORMER]
seeds = [42, 123, 54, 2345, 987, 6538]
algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel=SquaredExponential()).get_name(), UncertainRandomForest().get_name()]


minObs_dict = {}
regret_dict = {}
for dataset in datasets:
    algo_minObs = {}
    algo_regret = {}
    for a in algos:
        reps_minObs = {}
        reps_regret = {}
        for rep in representations:
            seed_minObs = []
            seed_regret = []
            for seed in seeds:
                exps = find_experiments_by_tags({EXPERIMENT_TYPE: OPTIMIZATION,
                                                DATASET: dataset, 
                                                METHOD: a, 
                                                REPRESENTATION: rep,
                                                SEED: str(seed)})
                assert len(exps) == 1, rep+a+dataset
                runs = mlflow.search_runs(experiment_ids=[exps[0].experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
                results = []
                for id in runs['run_id'].to_list():
                    for r in mlflow.tracking.MlflowClient().get_metric_history(id, OBSERVED_Y):
                        results.append(r.value)

                min_observed = [min(results[:i]) for i in range(1,len(results)+1)]
                seed_minObs.append(min_observed)

                _, Y = load_dataset(dataset, representation=rep)
                regret = [np.sum(results[:i])-np.min(Y) for i in range(1,len(results)+1)]
                seed_regret.append(regret)

            reps_minObs[rep] = seed_minObs
            reps_regret[rep] = seed_regret

        if a == 'GPsquared_exponential':
            a = "GPsqexp"
        algo_minObs[a] = reps_minObs
        algo_regret[a] = reps_regret

    minObs_dict[dataset] = algo_minObs
    regret_dict[dataset] = algo_regret        

plot_optimization_task(metric_values=minObs_dict, name='Best observed')
plot_optimization_task(metric_values=regret_dict, name='Regret')
