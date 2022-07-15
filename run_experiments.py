import numpy as np
from algorithm_factories import GPMaternFactory, get_key_for_factory, UncertainRFFactory, GPSEFactory, GPLinearFactory, KNNFactory, RandomForestFactory
from protocol_factories import RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory, BioSplitterFactory, FractionalSplitterFactory
from run_single_regression_task import run_single_regression_task
from util.mlflow.constants import TRANSFORMER, EVE, VAE, VAE_AUX, ONE_HOT, ESM
from util.mlflow.constants import LINEAR, NON_LINEAR, VAE_DENSITY, VAE_RAND, EVE_DENSITY
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT

datasets = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"] # "MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"
dim_reduction = LINEAR # LINEAR, NON_LINEAR
representations = [TRANSFORMER, ESM, ONE_HOT, EVE, EVE_DENSITY] # VAE_AUX, VAE_RAND, TRANSFORMER, VAE, ONE_HOT, ESM, EVE, VAE_AUX EXTRA 1D rep: VAE_DENSITY

# Protocols: RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory, BioSplitterFactory, FractionalSplitterFactory
protocol_factories = [RandomSplitterFactory] # TOXI randomsplitter all reps except eve
# protocol_factories = [PositionalSplitterFactory]
# protocol_factories = [BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 3, 3)]

# Methods: # KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory
method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory]] 

def run_experiments():
    for dataset in datasets:
        for representation in representations:
            for protocol_factory in protocol_factories:
                for protocol in protocol_factory:#(dataset):
                    for factory_key in method_factories:
                        print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: full")
                        run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                protocol=protocol, augmentation=None, dim=None, dim_reduction=dim_reduction, plot_cv=False)


def run_augmentation_experiments():
    for dataset in ["1FQG", "UBQT", "CALM"]: # "UBQT", "CALM", "1FQG"
        for representation in [TRANSFORMER, EVE, ONE_HOT, ESM]: # TRANSFORMER, VAE, ONE_HOT, ESM
            for dim in [2, 10, 100, 1000]:
                if representation == VAE and dim and int(dim) > 30:
                    if int(dim) > 30:
                        continue # skip, dimensions greater than original -> None
                if representation == EVE and dim and int(dim) > 50:
                    if int(dim) > 50:
                        continue
                for protocol_factory in [PositionalSplitterFactory]: # TODO: randomsplitter >1FQG dim=2
                    for protocol in protocol_factory(dataset):
                        for factory_key in method_factories:
                            for augmentation in [ROSETTA, EVE_DENSITY]: 
                                    print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: {dim}, aug: {augmentation}")
                                    run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                            protocol=protocol, augmentation=augmentation, dim=dim, dim_reduction=dim_reduction)
                

if __name__ == "__main__":
    # run_experiments()
    run_augmentation_experiments()
    
