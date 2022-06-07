import os
from os.path import join
from statistics import mean
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import mlflow
from tqdm import tqdm
from scipy.stats import spearmanr
from umap import UMAP
import warnings
from typing import Tuple

from algorithm_factories import ALGORITHM_REGISTRY
from data.load_dataset import load_dataset, get_alphabet
from data.train_test_split import AbstractTrainTestSplitter
from util.numpy_one_hot import numpy_one_hot_2dmat
from util.log_uncertainty import prep_for_logdict
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, MedSE, SEVar, MLL, SPEARMAN_RHO, REPRESENTATION, SPLIT, ONE_HOT
from util.mlflow.constants import GP_L_VAR, GP_LEN, GP_VAR, GP_MU, OPT_SUCCESS
from util.mlflow.constants import NON_LINEAR, LINEAR, GP_K_PRIOR, GP_D_PRIOR
from util.preprocess import scale_observations
from gpflow.utilities import print_summary
from gpflow import kernels
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


def run_single_regression_task(dataset: str, representation: str, method_key: str, train_test_splitter: AbstractTrainTestSplitter, augmentation: str, 
                                dim: int=None, dim_reduction=NON_LINEAR):
    method = ALGORITHM_REGISTRY[method_key](representation, get_alphabet(dataset))
    # load X for CV splitting
    X, Y = load_dataset(dataset, representation=ONE_HOT)
    seq_len = X.shape[1]
    train_indices, _, test_indices = train_test_splitter.split(X)
    X, Y = load_dataset(dataset, representation=representation)
    
    if representation is ONE_HOT:
        X = numpy_one_hot_2dmat(X, max=len(get_alphabet(dataset)))
        # normalize by sequence length
        X = X / seq_len

    tags = {DATASET: dataset, 
            METHOD: method.get_name(), 
            REPRESENTATION: representation,
            SPLIT: train_test_splitter.get_name(), 
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
            method.init_length = np.max(np.abs(np.subtract(X_train[0], X_train[1]))) # if reduced on lower dim this value is too small
        method.train(X_train, scaled_y)
        try:
            _mu, unc = method.predict_f(X_test)
        except tf.errors.InvalidArgumentError as _:
            warnings.warn(f"Experiment: {dataset}, rep: {representation} in d={dim} not stable, prediction failed!")
            _mu, unc = np.full(Y_test.shape, np.nan), np.full(Y_test.shape, np.nan)
        # undo scaling
        mu = _mu*std_y + mean_y
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

        mlflow.log_metric(MSE, mse, step=split)
        mlflow.log_metric(MedSE, medse, step=split)
        mlflow.log_metric(SEVar, mse_var, step=split)
        mlflow.log_metric(MLL, mll, step=split)
        mlflow.log_metric(SPEARMAN_RHO, r, step=split)
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


if __name__ == "__main__":
    dataset = "1FQG"
    from util.mlflow.constants import VAE, TRANSFORMER
    representation = TRANSFORMER
    from data.load_dataset import get_alphabet
    alphabet = get_alphabet(dataset)
    from gpflow.kernels import SquaredExponential
    from data.train_test_split import BlockPostionSplitter
    from algorithms.one_hot_gp import GPOneHotSequenceSpace
    from algorithms.gp_on_real_space import GPonRealSpace

    run_single_regression_task(dataset=dataset, representation=representation,
                               method=GPonRealSpace(kernel_factory=lambda: SquaredExponential()),
                               train_test_splitter=BlockPostionSplitter(dataset=dataset), augmentation=NO_AUGMENT)
