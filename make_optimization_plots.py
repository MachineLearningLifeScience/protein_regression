from typing import Tuple
import mlflow
import numpy as np
from gpflow.kernels import SquaredExponential, Matern52
from gpflow.kernels.linears import Linear
from mlflow.entities import ViewType
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from util.mlflow.constants import DATASET, METHOD, MSE, REPRESENTATION, STD_Y, TRANSFORMER, VAE, VAE_DENSITY, ESM
from util.mlflow.constants import SPLIT, ONE_HOT, NONSENSE, KNN_name, SEED, OPTIMIZATION, EXPERIMENT_TYPE, OBSERVED_Y
from util.mlflow.convenience_functions import get_mlflow_results_optimization
from data.load_dataset import load_dataset
from visualization.plot_metric_for_dataset import plot_optimization_task
from visualization.plot_metric_for_uncertainties import plot_uncertainty_optimization

# gathers all our results and saves them into a numpy array
datasets = ["TOXI"]
representations = [ESM] # TRANSFORMER
seeds = [11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245] # 11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245
#seeds = [11]

algos = [GPonRealSpace(kernel_factory=lambda: Matern52()).get_name(), 
        GPonRealSpace(kernel_factory=lambda: Linear()).get_name(), 
        UncertainRandomForest().get_name()]

results = get_mlflow_results_optimization(datasets=datasets, algos=algos, reps=representations, metrics=[OBSERVED_Y, STD_Y], seeds=seeds)
reference_results = get_mlflow_results_optimization(datasets=datasets, algos=[VAE_DENSITY], reps=[None], metrics=[OBSERVED_Y])

def compute_metrics_optimization_results(results: dict, datasets: list=datasets, algos: list=algos, representations: list=representations, seeds: list=seeds) -> Tuple[dict, dict, dict, dict]:
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
            if a == 'GPsquared_exponential':
                a = "GPsqexp"
            for rep in representations:
                seed_minObs = []
                seed_regret = []
                seed_meanObs = []
                seed_lastObs = []
                for seed in seeds:
                    _results = results[seed][dataset][a][rep][None][OBSERVED_Y]
                    min_observed = [min(_results[:i]) for i in range(1,len(_results)+1)]
                    seed_minObs.append(min_observed)
                    mean_observed = [np.mean(_results[:i]) for i in range(1,len(_results)+1)]
                    seed_meanObs.append(mean_observed)
                    last_observed = [_results[i] for i in range(0,len(_results)-1)]
                    seed_lastObs.append(last_observed)
                    _, Y = load_dataset(dataset, representation=ONE_HOT) # observations irrespective of representation
                    regret = [np.sum(_results[:i])-np.min(Y) for i in range(1,len(_results)+1)]
                    seed_regret.append(regret)
                reps_minObs[rep] = seed_minObs
                reps_regret[rep] = seed_regret
                reps_meanObs[rep] = seed_meanObs
                reps_lastObs[rep] = seed_lastObs
            algo_minObs[a] = reps_minObs
            algo_regret[a] = reps_regret
            algo_meanObs[a] = reps_meanObs
            algo_lastObs[a] = reps_lastObs
        minObs_dict[dataset] = algo_minObs
        regret_dict[dataset] = algo_regret 
        meanObs_dict[dataset] = algo_meanObs
        lastObs_dict[dataset] = algo_lastObs
    return minObs_dict, regret_dict, meanObs_dict, lastObs_dict
       
minObs_dict, regret_dict, meanObs_dict, lastObs_dict = compute_metrics_optimization_results(results)
ref_minObs_dict, ref_regret_dict, ref_meanObs_dict, ref_lastObs_dict = compute_metrics_optimization_results(reference_results, algos=[VAE_DENSITY], representations=[None], seeds=[None])
# add reference to results
minObs_dict[datasets[0]][VAE_DENSITY] = ref_minObs_dict[datasets[0]][VAE_DENSITY]
regret_dict[datasets[0]][VAE_DENSITY] = ref_regret_dict[datasets[0]][VAE_DENSITY]
meanObs_dict[datasets[0]][VAE_DENSITY] = ref_meanObs_dict[datasets[0]][VAE_DENSITY]
lastObs_dict[datasets[0]][VAE_DENSITY] = ref_lastObs_dict[datasets[0]][VAE_DENSITY]


plot_optimization_task(metric_values=minObs_dict, name=f'Best_observed_{representations}_{datasets}')
plot_optimization_task(metric_values=regret_dict, name=f'Regret_{representations}_{datasets}')
plot_optimization_task(metric_values=meanObs_dict, name=f'Mean_observed_{representations}_{datasets}')
plot_optimization_task(metric_values=lastObs_dict, name=f'Last_observed_{representations}_{datasets}')

plot_uncertainty_optimization(dataset=datasets[0], algos=[GPonRealSpace(kernel_factory=lambda: SquaredExponential()).get_name(), UncertainRandomForest().get_name()], 
                                rep=representations[0], seeds=seeds, number_quantiles=10, stepsize=2, min_obs_metrics=minObs_dict, regret_metrics=regret_dict)