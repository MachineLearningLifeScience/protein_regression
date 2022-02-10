from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from algorithms.mgp.fusion_scaler import BayesScaler
from algorithms.uncertain_rf import UncertainRandomForest
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from run_single_regression_task import run_single_regression_task
from run_single_regression_augmentation_task import run_single_augmentation_task
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT

datasets = ["1FQG"] # ["MTH3", "TIMB", "UBQT", "CALM", "BRCA"]
representations = [ONE_HOT, TRANSFORMER, VAE]
augmentations = [NO_AUGMENT]
train_test_splitters = [lambda dataset: RandomSplitter()] # [BlockPostionSplitter] 
optimize = True

if not optimize:
    import warnings
    warnings.warn("Optimization for GPs disabled.")

def RandomForestFactory(representation, alphabet):
    return RandomForest()

def KNNFactory(representation, alphabet):
    return KNN()

def BayesRegressorFactory(representation, alphabet):
    return BayesScaler()

def GPLinearFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), optimize=optimize)
    else:
        return GPonRealSpace(optimize=optimize)

def UncertainRFFactory(representation, alphabet):
    return UncertainRandomForest()

def GPSEFactory(representation, alphabet):
    if representation is ONE_HOT:
        return GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=SquaredExponential(), optimize=optimize)
    else:
        return GPonRealSpace(kernel=SquaredExponential(), optimize=optimize)

method_factories = [GPLinearFactory, UncertainRFFactory, GPSEFactory, RandomForestFactory, KNNFactory] #, BayesRegressorFactory]

def run_experiments():
    for dataset in datasets:
        for representation in representations:
            for train_test_splitter in train_test_splitters:
                alphabet = get_alphabet(dataset)
                for augmentation in augmentations:
                    for factory in method_factories:
                        method = factory(representation, alphabet)
                        run_single_regression_task(dataset=dataset, representation=representation, method=method,
                                                train_test_splitter=train_test_splitter(dataset=dataset),
                                                 augmentation=augmentation)


def run_augmentation_experiments():
    for dataset in ["1FQG"]: # ["UBQT", "CALM"]:
        for representation in [TRANSFORMER, ONE_HOT]:
            for train_test_splitter in train_test_splitters:
                alphabet = get_alphabet(dataset)
                for augmentation in [ROSETTA, VAE_DENSITY]:
                    for factory in [GPSEFactory, UncertainRFFactory]:
                        method = factory(representation, alphabet)
                        run_single_augmentation_task(dataset=dataset, representation=representation, method=method,
                                                train_test_splitter=train_test_splitter(dataset=dataset), augmentation=augmentation)
                

if __name__ == "__main__":
    run_experiments()
    run_augmentation_experiments()
