import os
from os.path import join
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_probability as tfp
import mlflow
from tqdm import tqdm
from scipy.stats import spearmanr
from umap import UMAP
import warnings
from typing import Tuple
from algorithm_factories import ALGORITHM_REGISTRY
from protocol_factories import PROTOCOL_REGISTRY
from data import load_dataset, get_alphabet, load_augmentation, get_load_function
from data.train_test_split import AbstractTrainTestSplitter, BioSplitter, PositionSplitter, WeightedTaskSplitter
from util.numpy_one_hot import numpy_one_hot_2dmat
from util.log_uncertainty import prep_for_logdict
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, MedSE, SEVar, MLL, SPEARMAN_RHO, REPRESENTATION, SPLIT, ONE_HOT, VAE_DENSITY
from util.mlflow.constants import GP_L_VAR, GP_LEN, GP_VAR, GP_MU, OPT_SUCCESS, NO_AUGMENT
from util.mlflow.constants import NON_LINEAR, LINEAR, GP_K_PRIOR, GP_D_PRIOR, MEAN_Y, STD_Y, PAC_BAYES_EPS
from util.preprocess import scale_observations
from bound.pac_bayes_bound import alquier_bounded_regression
from gpflow import kernels
from gpflow.utilities import to_default_float
from gpflow.models import GPR
from gpflow.utilities import print_summary
from gpflow.kernels.linears import Linear
from visualization.plot_training import plot_prediction_CV
mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("results", "mlruns")))


def _dim_reduce_X(dim: int, dim_reduction: str, X_train: np.ndarray, Y_train: np.ndarray) -> Tuple[np.ndarray, object]:
    """
    Dimensionality reduction on training split.
    Returns reduced training data and fitted dim_reduction.
    """
    if dim_reduction is NON_LINEAR:
        reducer = UMAP(n_components=dim, random_state=42, transform_seed=42)
    elif dim_reduction is LINEAR:
        reducer = PCA(n_components=dim)
    else:
        raise ValueError("Unspecified dimensionality reduction!")
    while X_train.shape[1] > dim:
        try:
            X_train = reducer.fit_transform(X_train, y=Y_train).astype(np.float64)
        except (TypeError, ValueError, np.linalg.LinAlgError) as _e:
            print(f"Dim reduction failed {dim} - retrying lower dim...")
            reducer.n_components = int(reducer.n_components - dim/10)
    return X_train, reducer


def run_single_regression_task(dataset: str, representation: str, method_key: str, protocol: AbstractTrainTestSplitter, 
                                augmentation: str, dim: int=None, dim_reduction=NON_LINEAR, plot_cv=False):
    method = ALGORITHM_REGISTRY[method_key](representation, get_alphabet(dataset))
    missed_assay_indices: np.ndarray = None
    # load X for CV splitting
    X, Y = load_dataset(dataset, representation=representation)
    seq_len = X.shape[1]

    if representation is ONE_HOT:
        X = numpy_one_hot_2dmat(X, max=len(get_alphabet(dataset)))
        # normalize by sequence length
        X = X / seq_len
        
    if augmentation: # add augmentation vector from Rosetta, EVE, VAE
        A, Y, missed_assay_indices = load_augmentation(name=dataset, augmentation=augmentation, representation=representation)
        if missed_assay_indices is not None and len(A) != len(X):
            X = np.delete(X, missed_assay_indices, axis=0) 
        A /= seq_len
        X = np.concatenate([X, A], axis=1)

    if type(protocol) in [PositionSplitter, BioSplitter]: # special case mismatches in position splitting
        if missed_assay_indices is not None:
            train_indices, _, test_indices = protocol.split(X, representation, missed_assay_indices)
        else:
            train_indices, _, test_indices = protocol.split(X, representation)
    elif type(protocol) == WeightedTaskSplitter:
        train_indices, _, test_indices = protocol.split(X, Y)
    else: # BASE CASE splitting:
        train_indices, _, test_indices = protocol.split(X)

    tags = {DATASET: dataset, 
            METHOD: method.get_name(), 
            REPRESENTATION: representation,
            SPLIT: protocol.get_name(), 
            AUGMENTATION: augmentation,
            "OPTIMIZE": method.optimize,
            }

    # record experiments by dataset name and have the tags as logged parameters
    _experiment = mlflow.set_experiment(dataset)
    mlflow.start_run()
    mlflow.set_tags(tags)

    for split in tqdm(range(0, len(train_indices))):
        X_train = X[train_indices[split], :]
        Y_train = Y[train_indices[split], :]
        X_test = X[test_indices[split], :]
        Y_test = Y[test_indices[split]]
        mean_y, std_y, scaled_y = scale_observations(Y_train)
        if dim and X_train.shape[1] > dim:
            X_train, reducer = _dim_reduce_X(dim=dim, dim_reduction=dim_reduction, X_train=X_train, Y_train=Y_train)
            X_test = reducer.transform(X_test).astype(np.float64)
            mlflow.set_tags({"DIM_REDUCTION": dim_reduction, "DIM": reducer.n_components})
        if "GP" in method.get_name() and not dim: # set initial parameters based on distance in space if using full latent space
            init_len = np.max(np.abs(np.subtract(X_train[0], X_train[1])))
            eps = 0.001
            method.init_length = init_len if init_len > 0.0 else init_len+eps # if reduced on lower dim this value is too small
        if "GP" in method.get_name():
            _, prior_cov = method.prior(X_train, scaled_y).predict_y(X_train)
        else: # uniform prior of observations std=1 in accordance with scaling
            _, prior_cov = np.zeros(X_train.shape[0])[:, np.newaxis], np.ones(X_train.shape[0])[:, np.newaxis]
        # TRAIN MODEL:
        method.train(X_train, scaled_y)
        try:
            _mu, _unc = method.predict_f(X_test)
            _, post_cov = method.predict(X_train)
        except tf.errors.InvalidArgumentError as _:
            warnings.warn(f"Experiment: {dataset}, rep: {representation} in d={dim} not stable, prediction failed!")
            _mu, _unc = np.full(Y_test.shape, np.nan), np.full(Y_test.shape, np.nan)
            post_cov = np.full(Y_train.shape, np.nan)
        # undo scaling
        mu = _mu*std_y + mean_y
        unc = _unc * std_y
        assert(mu.shape[1] == 1 == unc.shape[1])
        assert(mu.shape[0] == unc.shape[0] == len(test_indices[split]))
        # record mean and median smse and nll and spearman correlation
        assert mu.shape == Y_test.shape, "shape mismatch "+str(mu.shape)+' '+str(Y_test.flatten().shape)
        
        baseline = np.mean(np.square(Y_test - np.repeat(mean_y, len(Y_test)).reshape(-1,1)))
        err2 = np.square(Y_test - mu)
        mse = np.mean(err2)/baseline
        medse = np.median(err2)
        mse_var = np.var(err2)
        mll = np.mean(err2 / unc / 2 + np.log(2 * np.pi * unc) / 2)
        r = spearmanr(Y_test, mu)[0]  # we do not care about the p-value
        n = X_train.shape[0]
        unscaled_posterior_cov = np.array(post_cov)*std_y
        unscaled_prior_cov = np.array(prior_cov)*std_y
        pac_epsilon = alquier_bounded_regression(post_mu=np.zeros(n), post_cov=unscaled_posterior_cov, 
                                prior_mu=np.zeros(n), prior_cov=unscaled_prior_cov, n=n, delta=0.05, loss_bound=(0, 1))
        mlflow.log_metric(MSE, mse, step=split)
        mlflow.log_metric(MedSE, medse, step=split)
        mlflow.log_metric(SEVar, mse_var, step=split)
        mlflow.log_metric(MLL, mll, step=split)
        mlflow.log_metric(SPEARMAN_RHO, r, step=split)
        mlflow.log_metric(MEAN_Y, mean_y, step=split)  # record scaling information 
        mlflow.log_metric(STD_Y, std_y, step=split)
        mlflow.log_metric(PAC_BAYES_EPS, float(pac_epsilon), step=split)
        if "GP" in method.get_name():
            mlflow.log_metric(GP_VAR, float(method.gp.kernel.variance.numpy()), step=split)
            mlflow.log_metric(GP_L_VAR, float(method.gp.likelihood.variance.numpy()), step=split)
            if method.gp.kernel.__class__ != kernels.linears.Linear:
                mlflow.log_metric(GP_LEN, float(method.gp.kernel.lengthscales.numpy()), step=split)
                mlflow.set_tag(GP_K_PRIOR, method.gp.kernel.lengthscales.prior.name)
            mlflow.log_metric(OPT_SUCCESS, float(method.opt_success))
            mlflow.set_tag(GP_D_PRIOR, method.gp.likelihood.variance.prior.name)
        trues, mus, uncs, errs = prep_for_logdict(Y_test, mu, unc, err2, baseline)
        mlflow.log_dict({'trues': trues, 'pred': mus, 'unc': uncs, 'mse': errs}, 'split'+str(split)+'/output.json')
    mlflow.end_run()
