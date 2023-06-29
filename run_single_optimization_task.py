import os
from os.path import join
import numpy as np
import argparse
import mlflow
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import norm as normal
from util.preprocess import scale_observations
from util.log import prep_for_logdict
from algorithm_factories import ALGORITHM_REGISTRY
from data.load_dataset import load_dataset, get_alphabet
from util.mlflow.constants import DATASET, METHOD, ONE_HOT, REPRESENTATION, SEED, OPTIMIZATION, EXPERIMENT_TYPE, OBSERVED_Y, OPTIMIZATION, VAE_DENSITY, EVE_DENSITY
from util.mlflow.constants import GP_LEN, GP_L_VAR, GP_VAR, GP_D_PRIOR, GP_K_PRIOR, OPT_SUCCESS
from util.mlflow.constants import MSE, MedSE, SEVar, MLL, SPEARMAN_RHO, MEAN_Y, STD_Y, AT_RANDOM
from util.mlflow.convenience_functions import find_experiments_by_tags, make_experiment_name_from_tags
from scipy import stats
from gpflow.utilities import print_summary
from gpflow import kernels
from scipy.stats import spearmanr

mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("results", "mlruns")))


def _expected_improvement(mean, variance, eta):
    s = np.sqrt(variance)
    gamma = (eta - mean) / s
    assert(gamma.shape[1] == 1)
    return s * (gamma * normal.cdf(gamma) + normal.pdf(gamma))


def run_single_optimization_task(dataset: str, method_key: str, seed: int, representation: str, max_iterations: int, log_interval: int=10, optimize=True):
    if method_key == "uncertainRF":
        optimize = False # minimal delta w.r.t. optimization and substantial difference in overall compute time.
    method = ALGORITHM_REGISTRY[method_key](representation, get_alphabet(dataset), optimize)
    X, Y = load_dataset(dataset, representation=representation)
    np.random.seed(seed)
    p = np.random.permutation(X.shape[0])
    X = X[p, :]
    Y = Y[p, :]

    tags = {EXPERIMENT_TYPE: OPTIMIZATION, 
            DATASET: dataset, 
            METHOD: method.get_name(), 
            REPRESENTATION: representation,
            SEED: str(seed),
            OPTIMIZATION: method.optimize,}

    # record experiments by dataset name and have the tags as logged parameters
    experiment_name = dataset + "_optimization"
    __experiment = mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tags(tags)

    next = 0
    selected_X = []
    log_interval = np.arange(2, max_iterations, log_interval)
    for i in tqdm(range(1, max_iterations+1)):
        if i < 3 and "RF" in method.get_name():
            method.optimize = False # NOTE: RF optimization has internal 3-fold CV => base config for first iterations
        else:
            method.optimize = optimize # NOTE: we do not reinstantiate the method, reset to provided optimize parameter
        selected_X.append(next)
        # the .sum() is a hack to get a float value--mlflow complains about numpy arrays
        mlflow.log_metric(OBSERVED_Y, np.squeeze(Y[next, :]).sum(), step=i)
        X_observed = X[selected_X, :]
        Y_observed = Y[selected_X, :]
        # if enough data, scale observations
        if len(selected_X) > 1:
            mean_y, std_y, Y_observed = scale_observations(Y_observed) # outputs scaled_y as Y_observed
        method.train(X_observed, Y_observed)
        remaining_X = np.setdiff1d(np.arange(X.shape[0]), selected_X, assume_unique=True)
        remaining_Y = Y[remaining_X, :]
        candidates = X[remaining_X, :]
        _mu, _unc = method.predict_f(candidates)
        assert(_mu.shape[1] == 1 == _unc.shape[1])
        assert(_mu.shape[0] == _unc.shape[0] == candidates.shape[0])
        # we are minimizing
        eta = np.min(Y_observed, axis=0)
        scoring = _expected_improvement(_mu, _unc, eta)
        # the acquisition function we maximize
        best_candidate = np.argmax(scoring, axis=0)[0]
        next = remaining_X[best_candidate]

        Sharpness = np.std(_unc, ddof=1)/np.mean(_unc)
        mlflow.log_metric('Sharpness', Sharpness, step=i)
        EI_y_corr = stats.spearmanr(scoring,remaining_Y)[0]
        mlflow.log_metric('EI_y_corr', EI_y_corr, step=i)
        if i in log_interval:
            _log_optimization_metrics_to_mlflow(method=method, remaining_Y=remaining_Y, mean_y=mean_y, std_y=std_y,
                                                _mu=_mu, _unc=_unc, step=i)
    mlflow.end_run()


def _log_optimization_metrics_to_mlflow(method, remaining_Y, mean_y, std_y, _mu, _unc, step) -> None:
    """
    Compute metrics and write to registered mlflow experiment.
    """
    # undo scaling
    mu = _mu*std_y + mean_y
    unc = _unc*std_y
    # compute metrics
    baseline = np.mean(np.square(remaining_Y - np.repeat(mean_y, len(remaining_Y)).reshape(-1,1)))
    err2 = np.square(remaining_Y - mu)
    mse = np.mean(err2)/baseline
    medse = np.median(err2)
    mse_var = np.var(err2)
    mll = np.mean(err2 / unc / 2 + np.log(2 * np.pi * unc) / 2)
    r = spearmanr(remaining_Y, mu)[0]  # we do not care about the p-value
    mlflow.log_metric(MSE, mse, step=step)
    mlflow.log_metric(MedSE, medse, step=step)
    mlflow.log_metric(SEVar, mse_var, step=step)
    mlflow.log_metric(MLL, mll, step=step)
    mlflow.log_metric(SPEARMAN_RHO, r, step=step)
    mlflow.log_metric(MEAN_Y, mean_y, step=step)  # record scaling information 
    mlflow.log_metric(STD_Y, std_y, step=step)
    if "GP" in method.get_name():
        mlflow.log_metric(GP_VAR, float(method.gp.kernel.variance.numpy()), step=step)
        mlflow.log_metric(GP_L_VAR, float(method.gp.likelihood.variance.numpy()), step=step)
        if method.gp.kernel.__class__ != kernels.linears.Linear:
            mlflow.log_metric(GP_LEN, float(method.gp.kernel.lengthscales.numpy()), step=step)
            mlflow.set_tag(GP_K_PRIOR, method.gp.kernel.lengthscales.prior.name)
        mlflow.log_metric(OPT_SUCCESS, float(method.opt_success))
        mlflow.set_tag(GP_D_PRIOR, method.gp.likelihood.variance.prior.name)
    trues, mus, uncs, errs = prep_for_logdict(remaining_Y, mu, unc, err2, baseline)
    mlflow.log_dict({'trues': trues, 'pred': mus, 'unc': uncs, 'mse': errs}, 'split'+str(step)+'/output.json')


def run_ranked_reference_task(dataset, max_iterations=500, log_interval=1, seed=42, reference_task=EVE_DENSITY):
    """
    Reference task to optimize / rank by dELBO density score.
    No EI computation, but simple best scores ranking
    Possible reference tasks: VAE_DENSITY, EVE_DENSITY
    """
    np.random.seed(seed)
    _reference_rep = reference_task if reference_task != AT_RANDOM else ONE_HOT
    X, Y = load_dataset(dataset, representation=_reference_rep)
    tags = {EXPERIMENT_TYPE: OPTIMIZATION, 
            DATASET: dataset, 
            METHOD: reference_task,
            OPTIMIZATION: False,}
    if reference_task == AT_RANDOM: # base reference
        tags[SEED] = seed
        sorted_idx = np.random.permutation(X.shape[0])
    else: # sort by dELBO densities
        sorted_idx = np.argsort(X, axis=0)
    X = X[sorted_idx, :]
    Y = Y[sorted_idx, :]
    # record experiments by dataset name and have the tags as logged parameters
    experiment_name = dataset + "_optimization"
    __experiment = mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tags(tags)
    # walk over ranked data and observe Ys
    for i in tqdm(range(1, max_iterations+1)):
        # the .sum() is a hack to get a float value--mlflow complains about numpy arrays
        mlflow.log_metric(OBSERVED_Y, np.squeeze(Y[i-1, :]).sum(), step=i)
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-m", "--method_key", type=str)
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-r", "--representation", type=str)
    parser.add_argument("-i", "--max_iterations", type=int, default=500)
    args = parser.parse_args()
    run_single_optimization_task(**vars(args))
