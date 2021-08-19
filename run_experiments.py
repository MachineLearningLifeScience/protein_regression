from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from run_single_regression_task import run_single_regression_task
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT

datasets = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"]
representations = [VAE, TRANSFORMER, ONE_HOT]
train_test_splitters = [BlockPostionSplitter]
#train_test_splitters = [lambda dataset: RandomSplitter()]


def RandomForestFactory(representation, alphabet):
    return RandomForest()

def KNNFactory(representation, alphabet):
    return KNN()

# TODO: Sometimes the linesearch can enter numerically unstable parameter regions.
# TODO: It seems there is now easy way to just downscale the linesearch.
optimize = False
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


method_factories = [KNNFactory] #[RandomForestFactory, GPSEFactory, GPLinearFactory]
for dataset in datasets:
    for representation in representations:
        for train_test_splitter in train_test_splitters:
            alphabet = get_alphabet(dataset)
            for factory in method_factories:
                method = factory(representation, alphabet)
                run_single_regression_task(dataset=dataset, representation=representation, method=method,
                                           train_test_splitter=train_test_splitter(dataset=dataset))
