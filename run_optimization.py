from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import UncertainRandomForest
from algorithms.KNN import KNN
from data.load_dataset import get_alphabet
from run_single_optimization_task import run_single_optimization_task
from util.mlflow.constants import TRANSFORMER, ONE_HOT

datasets = ["1FQG"]
representations = [TRANSFORMER]
seeds = [78543, 3465, 43245]

def RandomForestFactory(representation, alphabet):
    return RandomForest()

def UncertainRFFactory(representation, alphabet):
    return UncertainRandomForest()

def KNNFactory(representation, alphabet):
    return KNN()

# TODO: set this to true again!
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
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=SquaredExponential(), optimize=optimize)
    else:
        return GPonRealSpace(kernel=SquaredExponential(), optimize=optimize)


method_factories = [GPSEFactory, GPLinearFactory, UncertainRFFactory]
for dataset in datasets:
    for seed in seeds:
        for representation in representations:
            alphabet = get_alphabet(dataset)
            for factory in method_factories:
                method = factory(representation, alphabet)
                run_single_optimization_task(dataset=dataset, representation=representation,
                                            method=method, seed=seed)