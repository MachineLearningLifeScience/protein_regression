from typing import Tuple
from numpy import ndarray, hstack
from algorithms import AbstractAlgorithm

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