from algorithm_factories import get_key_for_factory, UncertainRFFactory, GPSEFactory, GPLinearFactory, KNNFactory, RandomForestFactory
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from run_single_regression_task import run_single_regression_task
from run_single_regression_augmentation_task import run_single_augmentation_task
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT, ESM, LINEAR, NON_LINEAR
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT

PROBLEM_CASES = ["UBQT", "BRCA"] # Error: VAE breaks

datasets = ["CALM", "1FQG", "UBQT", "BRCA"] # "MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA" # TODO: TIMB d=1000;None
dimensions = [2, 10, 100, 1000] # 2, 10, 100, 1000
dim_reduction = NON_LINEAR # LINEAR, NON_LINEAR
representations = [ONE_HOT, TRANSFORMER, VAE, ESM] # TRANSFORMER, VAE, ONE_HOT, ESM 
augmentations = [NO_AUGMENT]
train_test_splitters = [RandomSplitter()] # [lambda dataset: BlockPostionSplitter(dataset)] # [RandomSplitter()] # [lambda dataset: BlockPostionSplitter(dataset)] # [BlockPostionSplitter, RandomSplitter]  #  

# BLAT, UBQT, BRCA: KNNFactory, UncertainRFFactory, 
method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory]] # GPSEFactory, RandomForestFactory, GPLinearFactory
def run_experiments():
    for dataset in datasets:
        for representation in representations:
            for dim in dimensions:
                if representation == VAE and dim and int(dim) > 30:
                    if int(dim) > 30:
                        continue # skip, dimensions greater than original -> None
                for train_test_splitter in train_test_splitters:
                    for augmentation in augmentations:
                        for factory_key in method_factories:
                            print(f"{dataset}: {representation} - {factory_key}, dim: {dim}")
                            run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                    train_test_splitter= train_test_splitter,# train_test_splitter(dataset=dataset),  #train_test_splitter,# 
                                                    augmentation=augmentation, dim=dim, dim_reduction=dim_reduction)


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
    
