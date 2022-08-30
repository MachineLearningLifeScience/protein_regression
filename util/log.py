import numpy as np
from typing import Tuple
from numpy import ndarray, hstack
from algorithms import AbstractAlgorithm
from data.load_dataset import get_wildtype_and_offset, load_dataset
from data import get_alphabet
from util.mlflow.constants import ONE_HOT


def prep_for_logdict(y, mu, unc, err2, baseline):
    if type(mu)==np.ndarray:
        trues = list(np.hstack(y))
        mus = list(np.hstack(mu))
        uncs = list(np.hstack(unc))
        errs = list(np.hstack(err2/baseline))
    else:
        trues = list(np.hstack(y))
        mus = list(np.hstack(mu.cpu().numpy()))
        uncs = list(np.hstack(unc.cpu().numpy()))
        errs = list(np.hstack(err2/baseline))
    return trues, mus, uncs, errs


def prep_from_mixture(method, X: ndarray, Y: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Query GMM for assignment of input, means, cov and weights for components
    """
    if "GM" not in method.get_name():
        raise RuntimeError(f"Mixture model required! model-type={type(method)}")
    complete_sample = hstack([X, Y])
    assignment_vector = method.model.gmm_.to_responsibilities(complete_sample)
    mixture_means = method.model.gmm_.means
    mixture_covariances = method.model.gmm_.covariances
    mixture_weights = method.model.gmm_.priors
    return assignment_vector, mixture_means, mixture_covariances, mixture_weights


def prep_for_mutation(dataset: str, test_idx:np.ndarray):
    """
    Parse mutations from datasets against WT.
    Return list of mutations in test-datset.
    """
    wt, _ = get_wildtype_and_offset(dataset)
    X, _ = load_dataset(dataset, representation=ONE_HOT)
    alphabet = {v: k for k, v in get_alphabet(dataset).items()}
    mutations = []
    for seq in X[test_idx,:]:
        m_idx = np.where(seq!=wt)[0]
        from_aa, to_aa = wt[m_idx], seq[m_idx]
        mutations.append(["".join([alphabet.get(wt_aa), str(idx+1), alphabet.get(m_aa)]) # adjust for +1 offset from zero index
                            for wt_aa, idx, m_aa in zip(from_aa, m_idx, to_aa)])
    return mutations