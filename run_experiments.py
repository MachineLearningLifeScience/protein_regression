from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from run_single_regression_task import run_single_regression_task
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT, NONSENSE

datasets = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"]
#datasets = ["1FQG"]
representations = [VAE, TRANSFORMER, ONE_HOT, NONSENSE]
train_test_splitters = [BlockPostionSplitter]
#train_test_splitters = [lambda dataset: RandomSplitter()]


def RandomForestFactory(representation, alphabet):
    return RandomForest()


# TODO: set this to true again!
optimize = False
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


method_factories = [RandomForestFactory, GPSEFactory, GPLinearFactory]
for dataset in datasets:
    for representation in representations:
        for train_test_splitter in train_test_splitters:
            alphabet = get_alphabet(dataset)
            for factory in method_factories:
                method = factory(representation, alphabet)
                run_single_regression_task(dataset=dataset, representation=representation, method=method,
                                           train_test_splitter=train_test_splitter(dataset=dataset))
