import numpy as np
import mlflow
from tqdm import tqdm
from scipy.stats import spearmanr
from algorithms.abstract_algorithm import AbstractAlgorithm
from util.preprocess import scale_observations
from data.load_dataset import load_dataset, get_alphabet
from util.numpy_one_hot import numpy_one_hot_2dmat
from data.load_augmentation import load_augmentation
from data.train_test_split import AbstractTrainTestSplitter
from util.log_uncertainty import prep_for_logdict
from algorithm_factories import ALGORITHM_REGISTRY
from util.mlflow.constants import DATASET, METHOD, MSE, ROSETTA, MedSE, SEVar, MLL, SPEARMAN_RHO, REPRESENTATION, SPLIT, ONE_HOT, AUGMENTATION
from util.mlflow.constants import GP_LEN, GP_VAR, GP_L_VAR, GP_MU


def run_single_augmentation_task(dataset: str, representation: str, method_key: AbstractAlgorithm, augmentation: str, train_test_splitter: AbstractTrainTestSplitter):
    if not representation: 
        raise ValueError("Representation required for data loading...")
    method = ALGORITHM_REGISTRY[method_key](representation, get_alphabet(dataset))
    A, Y, missed_assay_indices = load_augmentation(name=dataset, augmentation=augmentation)
    # load X for the CV splitting requiring label-encoding
    X, Y = load_dataset(dataset, representation=ONE_HOT)
    if missed_assay_indices is not None and len(A) != len(X):
        X = np.delete(X, missed_assay_indices, axis=0) 
    train_indices, val_indices, test_indices = train_test_splitter.split(X)
    seq_len = X.shape[1]

    X, Y = load_dataset(dataset, representation=representation)
    if missed_assay_indices is not None and len(A) != len(X):
        X = np.delete(X, missed_assay_indices, axis=0) 
    if representation is ONE_HOT:
        X = numpy_one_hot_2dmat(X, max=len(get_alphabet(dataset)))
        # normalize by sequence length
        X = X / seq_len
    
    X = np.concatenate([X, A], axis=1)
    tags = {DATASET: dataset, 
            METHOD: method.get_name(), 
            REPRESENTATION: representation,
            SPLIT: train_test_splitter.get_name(), 
            AUGMENTATION: augmentation,
            "OPTIMIZE": method.optimize}
    # record experiments by dataset name and have the tags as logged parameters
    experiment = mlflow.set_experiment(dataset)
    mlflow.start_run(experiment_id=experiment)
    mlflow.set_tags(tags)
    
    for split in tqdm(range(0, len(train_indices))):
        X_train = X[train_indices[split], :]
        Y_train = Y[train_indices[split], :]
        Y_test = Y[test_indices[split]]
        mean_y, std_y, scaled_y = scale_observations(Y_train)
        method.train(X_train, scaled_y)
        #print_summary(method.gp)
        _mu, unc = method.predict_f(X[test_indices[split], :])
        # undo scaling
        mu = _mu*std_y + mean_y
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
        if "GP" in method.get_name():
            #mlflow.log_metric(GP_MU, method.gp.mean_function.c.numpy()[0], step=split)
            mlflow.log_metric(GP_VAR, float(method.gp.kernel.variance.numpy()), step=split)
            mlflow.log_metric(GP_L_VAR, float(method.gp.likelihood.variance.numpy()), step=split)
            #mlflow.log_metric(GP_LEN, float(method.gp.kernel.lengthscales.numpy()), step=split)
        trues, mus, uncs, errs = prep_for_logdict(Y[test_indices[split], :], mu, unc, err2, baseline)
        mlflow.log_dict({'trues': trues, 'pred': mus, 'unc': uncs, 'mse': errs}, 'split'+str(split)+'/output.json')
    mlflow.end_run()


if __name__ == "__main__":
    dataset = "UBQT"
    from util.mlflow.constants import VAE
    representation = VAE
    from data.load_dataset import get_alphabet
    alphabet = get_alphabet(dataset)
    from gpflow.kernels import SquaredExponential
    from data.train_test_split import BlockPostionSplitter
    from algorithms.one_hot_gp import GPOneHotSequenceSpace
    from algorithms.gp_on_real_space import GPonRealSpace

    run_single_augmentation_task(dataset=dataset, representation=representation,
                               method=GPonRealSpace(kernel=SquaredExponential()),
                               train_test_splitter=BlockPostionSplitter(dataset=dataset), augmentation=ROSETTA)
