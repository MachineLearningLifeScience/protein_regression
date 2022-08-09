from typing import Callable
from gpflow.kernels import SquaredExponential, Linear, Polynomial, Matern52, MultioutputKernel
from algorithms.abstract_algorithm import AbstractAlgorithm
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.KNN import KNN
from algorithms.gmm_regression import GMMRegression
from util.mlflow.constants import ONE_HOT


def RandomForestFactory(representation, alphabet):
    return RandomForest()

def UncertainRFFactory(representation, alphabet):
    return UncertainRandomForest()

def KNNFactory(representation, alphabet):
    return KNN()


optimize = True
if not optimize:
    import warnings
    warnings.warn("Optimization for GPs disabled.")

def GPLinearFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), optimize=optimize)
    else:
        return GPonRealSpace(optimize=optimize)

def GPSEFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel_factory=lambda: SquaredExponential(), optimize=optimize)
    else:
        return GPonRealSpace(kernel_factory=lambda: SquaredExponential(), optimize=optimize)

def GPMaternFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel_factory=lambda: Matern52(), optimize=optimize)
    else:
        return GPonRealSpace(kernel_factory=lambda: Matern52(), optimize=optimize)

def GPMOFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel_factory=lambda: MultioutputKernel(), optimize=optimize)
    else:
        return GPonRealSpace(kernel_factory=lambda: MultioutputKernel(), optimize=optimize)


def GMMFactory(representation, alphabet, n_components=2):
    return GMMRegression(n_components)



def get_key_for_factory(f: Callable[[], AbstractAlgorithm]):
    return f.__name__


ALGORITHM_REGISTRY = {
    get_key_for_factory(f): f for f in [RandomForestFactory, UncertainRFFactory, KNNFactory, GPLinearFactory,
                                        GPSEFactory, GPMaternFactory, GMMFactory]
}

