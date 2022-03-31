from algorithm_factories import get_key_for_factory, UncertainRFFactory, GPSEFactory
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from run_single_regression_task import run_single_regression_task
from run_single_regression_augmentation_task import run_single_augmentation_task
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT

datasets = ["1FQG"] # ["MTH3", "TIMB", "UBQT", "CALM", "BRCA"]
representations = [TRANSFORMER]
augmentations = [NO_AUGMENT]
train_test_splitters = [lambda dataset: BlockPostionSplitter(dataset)] # [BlockPostionSplitter, RandomSplitter] 

method_factories = [get_key_for_factory(UncertainRFFactory)]#, GPSEFactory] #, BayesRegressorFactory]
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


augmententation_method_factories = [get_key_for_factory(f) for f in [GPSEFactory, UncertainRFFactory]]
def run_augmentation_experiments():
    for dataset in ["1FQG"]: # ["UBQT", "CALM"]:
        for representation in [TRANSFORMER, ONE_HOT]:
            for train_test_splitter in train_test_splitters:
                alphabet = get_alphabet(dataset)
                for augmentation in [ROSETTA, VAE_DENSITY]:
                    for factory in augmententation_method_factories:
                        method = factory(representation, alphabet)
                        run_single_augmentation_task(dataset=dataset, representation=representation, method=method,
                                                train_test_splitter=train_test_splitter(dataset=dataset), augmentation=augmentation)
                

if __name__ == "__main__":
    run_experiments()
    #run_augmentation_experiments()
