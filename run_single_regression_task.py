import numpy as np
import mlflow
from scipy.stats import spearmanr

from algorithms.abstract_algorithm import AbstractAlgorithm
from data.load_dataset import load_dataset
from data.train_test_split import AbstractTrainTestSplitter
from util.mlflow.constants import DATASET, METHOD, MSE, MedSE, SEVar, MLL, SPEARMAN_RHO, REPRESENTATION, SPLIT, ONE_HOT
from util.mlflow.convenience_functions import find_experiments_by_tags, make_experiment_name_from_tags


def run_single_regression_task(dataset: str, representation: str, method: AbstractAlgorithm, train_test_splitter: AbstractTrainTestSplitter):
    X, Y = load_dataset(dataset, representation=ONE_HOT)
    train_indices, val_indices, test_indices = train_test_splitter.split(X)
    if representation is not None:
        X, Y = load_dataset(dataset, representation=representation)

    tags = {DATASET: dataset, METHOD: method.get_name(), REPRESENTATION: representation,
            SPLIT: train_test_splitter.get_name()}

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

    mlflow.start_run()
    for split in range(0, len(train_indices)):
        method.train(X[train_indices[split], :], Y[train_indices[split], :])
        mu, unc = method.predict(X[test_indices[split], :])
        assert(mu.shape[1] == 1 == unc.shape[1])
        assert(mu.shape[0] == unc.shape[0] == len(test_indices[split]))
        # record mean and median smse and nll and spearman correlation
        assert mu.shape == Y[test_indices[split]].shape, "shape mismatch "+str(mu.shape)+' '+str(Y[test_indices[split]].flatten().shape)
        
        y_mu = np.mean(Y[train_indices[split], :])    
        baseline = np.mean(np.square(Y[test_indices[split]] - np.repeat(y_mu, len(Y[test_indices[split]])).reshape(-1,1)))
        err2 = np.square(Y[test_indices[split]] - mu)
        mse = np.mean(err2)/baseline
        medse = np.median(err2)
        mse_var = np.var(err2)
        mll = np.mean(err2 / unc / 2 + np.log(2 * np.pi * unc) / 2)
        r = spearmanr(Y[test_indices[split]], mu)[0]  # we do not care about the p-value
        mlflow.log_metric(MSE, mse, step=split)
        mlflow.log_metric(MedSE, medse, step=split)
        mlflow.log_metric(SEVar, mse_var, step=split)
        mlflow.log_metric(MLL, mll, step=split)
        mlflow.log_metric(SPEARMAN_RHO, r, step=split)
    mlflow.end_run()


if __name__ == "__main__":
    dataset = "BRCA"
    from util.mlflow.constants import VAE
    representation = VAE
    from data.load_dataset import get_alphabet
    alphabet = get_alphabet(dataset)
    from gpflow.kernels import SquaredExponential
    from data.train_test_split import BlockPostionSplitter
    from algorithms.one_hot_gp import GPOneHotSequenceSpace
    from algorithms.gp_on_real_space import GPonRealSpace

    run_single_regression_task(dataset=dataset, representation=representation,
                               method=GPonRealSpace(kernel=SquaredExponential()),
                               train_test_splitter=BlockPostionSplitter(dataset=dataset))
