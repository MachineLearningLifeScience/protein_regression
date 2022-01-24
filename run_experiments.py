from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from algorithms.mgp.fusion_scaler import BayesScaler
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from run_single_regression_task import run_single_regression_task
from run_single_regression_augmentation_task import run_single_augmentation_task
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT, NONSENSE
from util.mlflow.constants import VAE_DENSITY, ROSETTA

#datasets = ["MTH3", "TIMB", "UBQT", "1FQG", "CALM", "BRCA"]
datasets = ["BRCA"]
representations = [VAE, TRANSFORMER, ONE_HOT, NONSENSE]
augmentations = [VAE, ROSETTA]
train_test_splitters = [BlockPostionSplitter]
#
# train_test_splitters = [lambda dataset: RandomSplitter()]


def RandomForestFactory(representation, alphabet):
    return RandomForest()

def KNNFactory(representation, alphabet):
    return KNN()

def BayesRegressorFactory(representation, alphabet):
    return BayesScaler()

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

method_factories = [RandomForestFactory, GPSEFactory, GPLinearFactory, KNNFactory, BayesRegressorFactory]


def run_experiments():
    for dataset in datasets:
        for representation in representations:
            for train_test_splitter in train_test_splitters:
                alphabet = get_alphabet(dataset)
                for factory in method_factories:
                    method = factory(representation, alphabet)
                    run_single_regression_task(dataset=dataset, representation=representation, method=method,
                                            train_test_splitter=train_test_splitter(dataset=dataset))


def run_augmentation_experiments():
    for dataset in ["UBQT", "1FQG", "CALM"]:
        for representation in [ONE_HOT, TRANSFORMER]:
            for train_test_splitter in train_test_splitter:
                alphabet = get_alphabet(dataset)
                for augmentation in [VAE_DENSITY, ROSETTA]:
                    for factory in [RandomForestFactory, GPSEFactory, BayesRegressorFactory]:
                        method = factory(representation, alphabet)
                        run_single_augmentation_task(dataset=dataset, representation=representation, method=method,
                                                train_test_splitter=train_test_splitter(dataset=dataset), augmentation=augmentation)
                


if __name__ == "__main__":
    run_experiments()
    run_augmentation_experiments()
