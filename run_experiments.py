from itertools import product
from algorithm_factories import GPMaternFactory, get_key_for_factory, UncertainRFFactory, GPSEFactory, GPLinearFactory, KNNFactory, RandomForestFactory
from algorithm_factories import GMMFactory#, GPMOFactory
from protocol_factories import RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory
from protocol_factories import BioSplitterFactory, FractionalSplitterFactory, WeightedTaskSplitterFactory, WeightedTaskRegressSplitterFactory
from run_single_regression_task import run_single_regression_task
from util.mlflow.constants import TRANSFORMER, EVE, VAE, VAE_AUX, ONE_HOT, ESM
from util.mlflow.constants import LINEAR, NON_LINEAR, VAE_DENSITY, VAE_RAND, EVE_DENSITY
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT


datasets = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"] # "MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"
representations = [EVE_DENSITY, EVE, TRANSFORMER, ONE_HOT, ESM] # VAE_AUX, VAE_RAND, TRANSFORMER, VAE, ONE_HOT, ESM, EVE, VAE_AUX EXTRA 1D rep: VAE_DENSITY
MOCK = False
# Protocols: RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory, BioSplitterFactory, FractionalSplitterFactory
protocol_factories = [RandomSplitterFactory, PositionalSplitterFactory]
# protocol_factories = [FractionalSplitterFactory]
# protocol_factories = [WeightedTaskSplitterFactory]
# protocol_factories = [BioSplitterFactory("TOXI", 1, 2), BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4)]
# [BioSplitterFactory("TOXI", 1, 2), BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4)]:

# Methods: # KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory
# method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory]]
method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory]]

# TODO: rerun with KNN and RF for sanity check after data-load refactor:
experiment_iterator = product(datasets, representations, protocol_factories, method_factories)
def run_experiments():
    for dataset, representation, protocol_factory, factory_key in experiment_iterator:
        protocol_factory = protocol_factory(dataset) if type(protocol_factory) != list else protocol_factory
        for protocol in protocol_factory:
            print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: full")
            run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                    protocol=protocol, augmentation=None, dim=None, dim_reduction=LINEAR, mock=MOCK)

dim_reduction_experiment_iterator = product(["UBQT", "CALM", "1FQG"],
                                            [ONE_HOT, TRANSFORMER, EVE, ESM],
                                            [PositionalSplitterFactory, RandomSplitterFactory],
                                            method_factories,
                                            [LINEAR],
                                            [2, 10, 100, 1000])
def run_dim_reduction_experiments():
    for dataset, representation, protocol_factory, factory_key, dim_reduction, dim in dim_reduction_experiment_iterator:
        for protocol in protocol_factory(dataset):
            if representation == VAE and dim and int(dim) > 30:
                continue # skip, dimensions greater than original -> None
            if representation == EVE and dim and int(dim) > 50:
                continue
            print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: {dim} {dim_reduction}, aug: None")
            run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                        protocol=protocol, augmentation=None, dim=dim, dim_reduction=dim_reduction, mock=MOCK)


augmentation_experiment_iterator = product(["CALM", "UBQT", "1FQG"],
                                    [EVE, TRANSFORMER, ONE_HOT, ESM],
                                    [PositionalSplitterFactory, RandomSplitterFactory],
                                    method_factories,
                                    [ROSETTA, EVE_DENSITY])
def run_augmentation_experiments():
    for dataset, representation, protocol_factory, factory_key, augmentation in augmentation_experiment_iterator:
        for protocol in protocol_factory(dataset):
            print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: full, aug: {augmentation}")
            run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                        protocol=protocol, augmentation=augmentation, dim=None, dim_reduction=LINEAR, mock=MOCK)


# def run_threshold_experiments():
#     for t, dataset in zip([0., 0.], ["1FQG", "UBQT", "CALM"]):
#         for representation in [ESM, TRANSFORMER, ONE_HOT, EVE]:
#             for protocol_factory in [FractionalSplitterFactory]: # [RandomSplitterFactory, PositionalSplitterFactory] #[BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4)]:
#                 for protocol in protocol_factory(dataset):
#                     for factory_key in [get_key_for_factory(f) for f in [GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory]]:
#                         print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()}")
#                         run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
#                                                     protocol=protocol, augmentation=None, dim=None, threshold=t)


if __name__ == "__main__":
    # run_experiments()
    # ABLATION STUDY: (dim-reduction, augmentation, threshold):
    # run_dim_reduction_experiments()
    run_augmentation_experiments()
    #run_threshold_experiments()
    
