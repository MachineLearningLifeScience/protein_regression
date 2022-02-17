import numpy as np
import mlflow
from tqdm import tqdm
from scipy.stats import spearmanr

from algorithms.abstract_algorithm import AbstractAlgorithm
from data.load_dataset import load_dataset, get_alphabet
from data.train_test_split import AbstractTrainTestSplitter
from util.numpy_one_hot import numpy_one_hot_2dmat
from util.log_uncertainty import prep_for_logdict
from util.mlflow.constants import AUGMENTATION, DATASET, METHOD, MSE, MedSE, SEVar, MLL, SPEARMAN_RHO, REPRESENTATION, SPLIT, ONE_HOT

def run_single_regression_task(dataset: str, representation: str, method: AbstractAlgorithm, train_test_splitter: AbstractTrainTestSplitter, augmentation: str):
    # load X for CV splitting
    X, Y = load_dataset(dataset, representation=ONE_HOT)
    train_indices, val_indices, test_indices = train_test_splitter.split(X)
    X, Y = load_dataset(dataset, representation=representation)
    
    if representation is ONE_HOT:
        X = numpy_one_hot_2dmat(X, max=len(get_alphabet(dataset)))

    tags = {DATASET: dataset, 
            METHOD: method.get_name(), 
            REPRESENTATION: representation,
            SPLIT: train_test_splitter.get_name(), 
            AUGMENTATION: augmentation}

    # record experiments by dataset name and have the tags as logged parameters
    experiment = mlflow.set_experiment(dataset)
    mlflow.start_run()
    mlflow.set_tags(tags)

    for split in tqdm(range(0, len(train_indices))):
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
        trues, mus, uncs, errs = prep_for_logdict(Y[test_indices[split], :], mu, unc, err2, baseline)
        mlflow.log_dict({'trues': trues, 'pred': mus, 'unc': uncs, 'mse': errs}, 'split'+str(split)+'/output.json')
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
