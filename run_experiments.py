from algorithm_factories import GPMaternFactory, get_key_for_factory, UncertainRFFactory, GPSEFactory, GPLinearFactory, KNNFactory, RandomForestFactory
from algorithm_factories import GMMFactory#, GPMOFactory
from protocol_factories import RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory
from protocol_factories import BioSplitterFactory, FractionalSplitterFactory, WeightedTaskSplitterFactory, WeightedTaskRegressSplitterFactory
from run_single_regression_task import run_single_regression_task
from util.mlflow.constants import TRANSFORMER, EVE, VAE, VAE_AUX, ONE_HOT, ESM
from util.mlflow.constants import LINEAR, NON_LINEAR, VAE_DENSITY, VAE_RAND, EVE_DENSITY
from util.mlflow.constants import VAE_DENSITY, ROSETTA, NO_AUGMENT

datasets = ["TOXI"] # "MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"
representations = [EVE] # VAE_AUX, VAE_RAND, TRANSFORMER, VAE, ONE_HOT, ESM, EVE, VAE_AUX EXTRA 1D rep: VAE_DENSITY

# Protocols: RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory, BioSplitterFactory, FractionalSplitterFactory
protocol_factories = [RandomSplitterFactory]
# protocol_factories = [PositionalSplitterFactory]
# protocol_factories = [FractionalSplitterFactory]
# protocol_factories = [WeightedTaskSplitterFactory]
# protocol_factories = [WeightedTaskRegressSplitterFactory] # TODO: test against all proteins on functional threshold
# protocol_factories = [BioSplitterFactory("TOXI", 1, 2), BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4)]
# [BioSplitterFactory("TOXI", 1, 2), BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4)]:

# Methods: # KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory
method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory]]


def run_experiments():
    for dataset in datasets:
        for representation in representations:
            for protocol_factory in protocol_factories:
                protocol_factory = protocol_factory(dataset) if type(protocol_factory) != list else protocol_factory
                for protocol in protocol_factory:
                    for factory_key in method_factories:
                        print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: full")
                        run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                protocol=protocol, augmentation=None, dim=None, dim_reduction=LINEAR, plot_cv=False)


def run_dim_reduction_experiments():
    for dataset in ["UBQT", "CALM", "1FQG"]: #["TOXI"]:
        for representation in [ONE_HOT]: # TRANSFORMER, EVE, ONE_HOT, ESM
            for protocol_factory in [PositionalSplitterFactory, RandomSplitterFactory]:
                for protocol in protocol_factory(dataset):
                    for factory_key in method_factories:
                        for dim_reduction in [LINEAR]: # LINEAR, NON_LINEAR
                            for dim in [2, 10, 100, 1000]: # 2, 10, 100, 1000
                                if representation == VAE and dim and int(dim) > 30:
                                    continue # skip, dimensions greater than original -> None
                                if representation == EVE and dim and int(dim) > 50:
                                    continue
                                print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: {dim} {dim_reduction}, aug: None")
                                run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                            protocol=protocol, augmentation=None, dim=dim, dim_reduction=dim_reduction)


def run_augmentation_experiments():
    for dataset in ["UBQT"]: # "UBQT", "CALM", "1FQG" # TODO: BLAT eve, UBQT, eve
        for representation in [TRANSFORMER, ONE_HOT, ESM, EVE]: # TRANSFORMER, VAE, ONE_HOT, ESM, EVE
                for protocol_factory in [PositionalSplitterFactory, RandomSplitterFactory]:
                    for protocol in protocol_factory(dataset):
                        for factory_key in method_factories:
                            for augmentation in [EVE_DENSITY, ROSETTA]:
                                print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: full, aug: {augmentation}")
                                run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                            protocol=protocol, augmentation=augmentation, dim=None, dim_reduction=LINEAR)


def run_threshold_experiments():
    for t, dataset in zip([0., 0.], ["1FQG", "UBQT", "CALM"]):
        for representation in [ESM, TRANSFORMER, ONE_HOT, EVE]:
            for protocol_factory in [FractionalSplitterFactory]: # [RandomSplitterFactory, PositionalSplitterFactory] #[BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4)]:
                for protocol in protocol_factory(dataset):
                    for factory_key in [get_key_for_factory(f) for f in [GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory]]:
                        print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()}")
                        run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                    protocol=protocol, augmentation=None, dim=None, threshold=t)


if __name__ == "__main__":
    # run_experiments()
    # ABLATION STUDY: (dim-reduction, augmentation, threshold):
    # run_dim_reduction_experiments()
    # run_augmentation_experiments()
    run_threshold_experiments()
    
