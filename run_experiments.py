from algorithm_factories import GPMaternFactory, get_key_for_factory, UncertainRFFactory, GPSEFactory, GPLinearFactory, KNNFactory, RandomForestFactory
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import BlockPostionSplitter, RandomSplitter, BioSplitter, PositionSplitter
from run_single_regression_task import run_single_regression_task
from run_single_regression_augmentation_task import run_single_augmentation_task
from util.mlflow.constants import TRANSFORMER, VAE, VAE_AUX, ONE_HOT, ESM, LINEAR, NON_LINEAR, VAE_DENSITY
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT

PROBLEM_CASES = ["UBQT", "BRCA"] # Error: VAE breaks

# TODO: MTH3 1000, None
datasets = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA"] # "MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"
dimensions = [None] # 2, 10, 100, 1000, None 
dim_reduction = LINEAR # LINEAR, NON_LINEAR
representations = [TRANSFORMER, VAE, ONE_HOT, ESM] # TRANSFORMER, VAE, ONE_HOT, ESM, # VAE_AUX EXTRA 1D rep: VAE_DENSITY
augmentations = [None]
train_test_splitters = [lambda dataset: PositionSplitter(dataset, 15)] # [lambda dataset: BioSplitter(dataset)] # [lambda dataset: BlockPostionSplitter(dataset)] # [RandomSplitter()] # 


# Methods: # KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory
method_factories = [get_key_for_factory(f) for f in [GPSEFactory]] 
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
                                                    train_test_splitter= train_test_splitter(dataset=dataset),  # train_test_splitter,# 
                                                    augmentation=augmentation, dim=dim, dim_reduction=dim_reduction, plot_cv=False)


augmententation_method_factories = [get_key_for_factory(f) for f in [GPSEFactory, RandomForestFactory, UncertainRFFactory, GPLinearFactory]] # TODO uncertainrandomforest
def run_augmentation_experiments():
    for dataset in ["1FQG"]:#, "UBQT", "CALM", "1FQG"
        for representation in [ESM]: # TRANSFORMER, VAE, ONE_HOT, ESM
            for dim in [10, 100, 1000, None]: # 2, 10, 100, 1000
                if representation == VAE and dim and int(dim) > 30:
                    if int(dim) > 30:
                        continue # skip, dimensions greater than original -> None
                for train_test_splitter in train_test_splitters:
                    for augmentation in [ROSETTA, VAE_DENSITY]: 
                        for factory_key in augmententation_method_factories:
                            print(f"{dataset}: {representation} - {factory_key} - aug: {augmentation}, d={dim}")
                            run_single_augmentation_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                    train_test_splitter=train_test_splitter,#  train_test_splitter(dataset=dataset), #  train_test_splitter, # 
                                                    augmentation=augmentation, dim=dim, dim_reduction=dim_reduction)
                

if __name__ == "__main__":
    run_experiments()
    #run_augmentation_experiments() #TODO: blocksplitter all
    
