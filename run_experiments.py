from pickletools import optimize
from algorithm_factories import get_key_for_factory, UncertainRFFactory, GPSEFactory, GPLinearFactory, KNNFactory, RandomForestFactory
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from run_single_regression_task import run_single_regression_task
from run_single_regression_augmentation_task import run_single_augmentation_task
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT

PROBLEM_CASES = ["UBQT", "BRCA"] # Error: VAE breaks

datasets = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA"] # "MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA"
dimensions = [2, 10, 100, 1000, None] # TODO delete and rerun all experiments for MTH3
# dimensions = [None]
representations = [TRANSFORMER, VAE, ONE_HOT] # TRANSFORMER, VAE, ONE_HOT
augmentations = [NO_AUGMENT]
train_test_splitters = [RandomSplitter()] # [lambda dataset: BlockPostionSplitter(dataset)] # [RandomSplitter()] # [lambda dataset: BlockPostionSplitter(dataset)] # [BlockPostionSplitter, RandomSplitter]  #  

method_factories = [get_key_for_factory(f) for f in [KNNFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory]] #, BayesRegressorFactory] GPSEFactory, RandomForestFactory, GPLinearFactory
def run_experiments():
    for dataset in datasets:
        for representation in representations:
            for dim in dimensions:
                for train_test_splitter in train_test_splitters:
                    for augmentation in augmentations:
                        for factory_key in method_factories:
                            print(f"{dataset}: {representation} - {factory_key}, dim: {dim}")
                            run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                    train_test_splitter= train_test_splitter,# train_test_splitter(dataset=dataset),  #
                                                    augmentation=augmentation, dim=dim)


augmententation_method_factories = [get_key_for_factory(f) for f in [GPSEFactory, RandomForestFactory, GPLinearFactory]]
def run_augmentation_experiments():
    for dataset in ["UBQT"]:#, "UBQT", "CALM", "1FQG"
        print(dataset)
        for representation in [ONE_HOT]: # TRANSFORMER, VAE, ONE_HOT
            for train_test_splitter in train_test_splitters:
                for augmentation in [ROSETTA, VAE_DENSITY]:
                    for factory_key in augmententation_method_factories:
                        print(f"{dataset}: {representation} - {factory_key}")
                        run_single_augmentation_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                train_test_splitter= train_test_splitter(dataset=dataset), #  train_test_splitter, #
                                                augmentation=augmentation)
                

if __name__ == "__main__":
    run_experiments()
    #run_augmentation_experiments()
    
