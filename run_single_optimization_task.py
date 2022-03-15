import numpy as np
import mlflow
import warnings
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import norm as normal
from algorithms.abstract_algorithm import AbstractAlgorithm
from gpflow.kernels import SquaredExponential
from algorithms.gp_on_real_space import GPonRealSpace
from data.load_dataset import load_dataset, get_alphabet
from util.mlflow.constants import DATASET, METHOD, REPRESENTATION, TRANSFORMER, SEED, OPTIMIZATION, EXPERIMENT_TYPE, \
    OBSERVED_Y, VAE, ONE_HOT
from util.mlflow.convenience_functions import find_experiments_by_tags, make_experiment_name_from_tags
from scipy import stats

def _expected_improvement(mean, variance, eta):
    s = np.sqrt(variance)
    gamma = (eta - mean) / s
    assert(gamma.shape[1] == 1)
    return s * (gamma * normal.cdf(gamma) + normal.pdf(gamma))


def run_single_optimization_task(dataset: str, method: AbstractAlgorithm, seed: int, representation=TRANSFORMER,
                                 max_iterations=500):
    X, Y = load_dataset(dataset, representation=representation)
    # TODO: find out which datasets are minimization and which are maximiation problems
    np.random.seed(seed)
    p = np.random.permutation(X.shape[0])
    X = X[p, :]
    Y = Y[p, :]

    tags = {EXPERIMENT_TYPE: OPTIMIZATION, 
            DATASET: dataset, 
            METHOD: method.get_name(), 
            REPRESENTATION: representation,
            SEED: str(seed)}

    exp = find_experiments_by_tags(tags)
    if len(exp) == 0:
        experiment_name = make_experiment_name_from_tags(tags)
        e_id = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        for t in tags.keys():
            mlflow.tracking.MlflowClient().set_experiment_tag(e_id, t, tags[t])
    elif len(exp) == 1:
        mlflow.set_experiment(exp[0].name)
    else:
        raise RuntimeError("There should be at most one experiment for a given tag combination!")

    next = 0
    selected_X = []
    mlflow.start_run()
    for i in tqdm(range(1, max_iterations+1)):
        selected_X.append(next)
        # the .sum() is a hack to get a float value--mlflow complains about numpy arrays
        mlflow.log_metric(OBSERVED_Y, np.squeeze(Y[next, :]).sum(), step=i)
        X_observed = X[selected_X, :]
        Y_observed = Y[selected_X, :]
        method.train(X_observed, Y_observed)
        remaining_X = np.setdiff1d(np.arange(X.shape[0]), selected_X, assume_unique=True)
        remaining_Y = Y[remaining_X, :]
        candidates = X[remaining_X, :]
        mu, unc = method.predict_f(candidates)
        assert(mu.shape[1] == 1 == unc.shape[1])
        assert(mu.shape[0] == unc.shape[0] == candidates.shape[0])
        # we are minimizing
        eta = np.min(Y_observed, axis=0)
        scoring = _expected_improvement(mu, unc, eta)
        # the acquisition function we maximize
        best_candidate = np.argmax(scoring, axis=0)[0]
        next = remaining_X[best_candidate]

        Sharpness = np.std(unc, ddof=1)/np.mean(unc)
        mlflow.log_metric('Sharpness', Sharpness, step=i)
        EI_y_corr = stats.spearmanr(scoring,remaining_Y)[0]
        print('a', np.mean(scoring), np.max(unc), np.min(unc), np.mean(EI_y_corr))

        mlflow.log_metric('EI_y_corr', EI_y_corr, step=i)

    mlflow.end_run()


if __name__ == "__main__":
    dataset = "BRCA"
    representation = VAE
    alphabet = get_alphabet(dataset)

    run_single_optimization_task(dataset=dataset, representation=representation,
                                 method=GPonRealSpace(kernel=SquaredExponential()), seed=0)
