import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential, Matern52
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
seeds = [123, 54, 2345, 987, 6538, 78543, 3465, 43245]
#seeds = [11]

algos = [GPonRealSpace(kernel_factory=lambda: SquaredExponential()).get_name(), 
        UncertainRandomForest().get_name()]
#   

minObs_dict = {}
regret_dict = {}
meanObs_dict = {}
lastObs_dict = {}
for dataset in datasets:
    algo_minObs = {}
    algo_regret = {}
    algo_meanObs = {}
    algo_lastObs = {}
    for a in algos:
        reps_minObs = {}
        reps_regret = {}
        reps_meanObs = {}
        reps_lastObs = {}
        for rep in representations:
            seed_minObs = []
            seed_regret = []
            seed_meanObs = []
            seed_lastObs = []
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

                mean_observed = [np.mean(results[:i]) for i in range(1,len(results)+1)]
                seed_meanObs.append(mean_observed)

                last_observed = [results[i] for i in range(0,len(results)-1)]
                seed_lastObs.append(last_observed)

                _, Y = load_dataset(dataset, representation=rep)
                regret = [np.sum(results[:i])-np.min(Y) for i in range(1,len(results)+1)]
                seed_regret.append(regret)

            reps_minObs[rep] = seed_minObs
            reps_regret[rep] = seed_regret
            reps_meanObs[rep] = seed_meanObs
            reps_lastObs[rep] = seed_lastObs

        if a == 'GPsquared_exponential':
            a = "GPsqexp"
        algo_minObs[a] = reps_minObs
        algo_regret[a] = reps_regret
        algo_meanObs[a] = reps_meanObs
        algo_lastObs[a] = reps_lastObs

    minObs_dict[dataset] = algo_minObs
    regret_dict[dataset] = algo_regret 
    meanObs_dict[dataset] = algo_meanObs
    lastObs_dict[dataset] = algo_lastObs        

plot_optimization_task(metric_values=minObs_dict, name=f'Best_observed_{representations}_{datasets}')
plot_optimization_task(metric_values=regret_dict, name=f'Regret_{representations}_{datasets}')
plot_optimization_task(metric_values=meanObs_dict, name=f'Mean_observed_{representations}_{datasets}')
plot_optimization_task(metric_values=lastObs_dict, name=f'Last_observed_{representations}_{datasets}')