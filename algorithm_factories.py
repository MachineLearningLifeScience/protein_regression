import warnings
from typing import Callable
from gpflow.kernels import SquaredExponential, Linear, Polynomial, Matern52, LinearCoregionalization
from algorithms.abstract_algorithm import AbstractAlgorithm
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.KNN import KNN
from algorithms.gmm_regression import GMMRegression
from util.mlflow.constants import ONE_HOT


OPTIMIZE = True
if not OPTIMIZE:
    warnings.warn("Optimization for Algorithms disabled.")


def RandomForestFactory(representation, alphabet):
    return RandomForest(optimize=OPTIMIZE)


def UncertainRFFactory(representation, alphabet):
    return UncertainRandomForest(optimize=OPTIMIZE)


def KNNFactory(representation, alphabet):
    return KNN(optimize=OPTIMIZE)


def GPLinearFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), optimize=OPTIMIZE)
    else:
        return GPonRealSpace(optimize=OPTIMIZE)


def GPSEFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel_factory=lambda: SquaredExponential(), optimize=OPTIMIZE)
    else:
        return GPonRealSpace(kernel_factory=lambda: SquaredExponential(), optimize=OPTIMIZE)


def GPMaternFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel_factory=lambda: Matern52(), optimize=OPTIMIZE)
    else:
        return GPonRealSpace(kernel_factory=lambda: Matern52(), optimize=OPTIMIZE)


def GPLinearRegionFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel_factory=lambda: LinearCoregionalization(kernels=[Linear(0), Linear(1)]), optimize=OPTIMIZE)
    else:
        return GPonRealSpace(kernel_factory=lambda: LinearCoregionalization(kernels=[Linear(0), Linear(1)]), optimize=OPTIMIZE)


def GMMFactory(representation, alphabet, n_components=2):
    return GMMRegression(n_components)



def get_key_for_factory(f: Callable[[], AbstractAlgorithm]):
    return f.__name__


ALGORITHM_REGISTRY = {
    get_key_for_factory(f): f for f in [RandomForestFactory, UncertainRFFactory, KNNFactory, GPLinearFactory,
                                        GPSEFactory, GPMaternFactory, GMMFactory]
}

