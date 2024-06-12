import warnings
from typing import Callable

from gpflow.kernels import (
    Linear,
    LinearCoregionalization,
    Matern52,
    Polynomial,
    SquaredExponential,
)

from algorithms.abstract_algorithm import AbstractAlgorithm
from algorithms.gmm_regression import GMMRegression
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.KNN import KNN
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import UncertainRandomForest
from util.mlflow.constants import ONE_HOT


def RandomForestFactory(representation, alphabet, optimize):
    return RandomForest(optimize=optimize)


def UncertainRFFactory(representation, alphabet, optimize):
    return UncertainRandomForest(optimize=optimize)


def KNNFactory(representation, alphabet, optimize):
    return KNN(optimize=optimize)


def GPLinearFactory(representation, alphabet, optimize):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), optimize=optimize)
    else:
        return GPonRealSpace(optimize=optimize)


def GPSEFactory(representation, alphabet, optimize):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(
            alphabet_size=len(alphabet),
            kernel_factory=lambda: SquaredExponential(),
            optimize=optimize,
        )
    else:
        return GPonRealSpace(
            kernel_factory=lambda: SquaredExponential(), optimize=optimize
        )


def GPMaternFactory(representation, alphabet, optimize):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(
            alphabet_size=len(alphabet),
            kernel_factory=lambda: Matern52(),
            optimize=optimize,
        )
    else:
        return GPonRealSpace(kernel_factory=lambda: Matern52(), optimize=optimize)


def GPLinearRegionFactory(representation, alphabet, optimize):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(
            alphabet_size=len(alphabet),
            kernel_factory=lambda: LinearCoregionalization(
                kernels=[Linear(0), Linear(1)]
            ),
            optimize=optimize,
        )
    else:
        return GPonRealSpace(
            kernel_factory=lambda: LinearCoregionalization(
                kernels=[Linear(0), Linear(1)]
            ),
            optimize=optimize,
        )


def GMMFactory(representation, alphabet, optimize, n_components=2):
    return GMMRegression(n_components)


def get_key_for_factory(f: Callable[[], AbstractAlgorithm]):
    return f.__name__


ALGORITHM_REGISTRY = {
    get_key_for_factory(f): f
    for f in [
        RandomForestFactory,
        UncertainRFFactory,
        KNNFactory,
        GPLinearFactory,
        GPSEFactory,
        GPMaternFactory,
        GMMFactory,
    ]
}
