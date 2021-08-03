from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter
from run_single_regression_task import run_single_regression_task
from util.mlflow.constants import TRANSFORMER, VAE

datasets = ["1FQG"]
datasets = ["BRCA"]
#datasets = ["CALM"]
representations = [VAE, None, TRANSFORMER]
train_test_splitters = [BlockPostionSplitter]


def RandomForestFactory(representation, alphabet, wt):
    return RandomForest()


# TODO: Sometimes the linesearch can enter numerically unstable parameter regions.
# TODO: It seems there is now easy way to just downscale the linesearch.
optimize = True
def GPLinearFactory(representation, alphabet, wt):
    if representation is None:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), optimize=optimize)
    else:
        return GPonRealSpace(optimize=optimize)


def GPSEFactory(representation, alphabet, wt):
    if representation is None:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=SquaredExponential(), optimize=optimize)
    else:
        return GPonRealSpace(kernel=SquaredExponential(), optimize=optimize)


method_factories = [GPSEFactory, RandomForestFactory, GPLinearFactory]
for dataset in datasets:
    for representation in representations:
        for train_test_splitter in train_test_splitters:
            alphabet = get_alphabet(dataset)
            wt = get_wildtype(dataset)
            for factory in method_factories:
                method = factory(representation, alphabet, wt)
                run_single_regression_task(dataset=dataset, representation=representation, method=method,
                                           train_test_splitter=train_test_splitter(dataset=dataset))
