import argparse
from itertools import product
from pickletools import optimize
from typing import List

from algorithm_factories import (GPLinearFactory, GPMaternFactory, GPSEFactory,
                                 KNNFactory, RandomForestFactory,
                                 UncertainRFFactory, get_key_for_factory)
from protocol_factories import (BioSplitterFactory, FractionalSplitterFactory,
                                PositionalSplitterFactory,
                                RandomSplitterFactory)
from run_single_regression_task import run_single_regression_task
from util.mlflow.constants import (ESM, ESM1V, ESM2, EVE, EVE_DENSITY, LINEAR,
                                   ONE_HOT, PROTT5, PSSM, ROSETTA, TRANSFORMER,
                                   VAE)

datasets = ["MTH3", "TIMB", "CALM", "1FQG", "BRCA", "TOXI", "UBQT"]
representations = [TRANSFORMER, ESM, EVE, EVE_DENSITY, ONE_HOT, ESM1V, ESM2, PROTT5, PSSM]
# Protocols: RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory, BioSplitterFactory, FractionalSplitterFactory
protocol_factories = [RandomSplitterFactory, PositionalSplitterFactory, FractionalSplitterFactory]
biosplitter_factories = [BioSplitterFactory("TOXI", 1, 1), BioSplitterFactory("TOXI", 1, 2), BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4), BioSplitterFactory("TOXI", 4, 4)]
ablation_protocols_random_cv = [RandomSplitterFactory("TIMB", k=i) for i in [2, 3, 5, 7, 15]] + [PositionalSplitterFactory("TIMB", positions=p) for p in [5,10,20,25]] 

# Methods: # KNNFactory, RandomForestFactory, UncertainRFFactory, GPSEFactory, GPLinearFactory, GPMaternFactory
method_factories = [get_key_for_factory(f) for f in [KNNFactory, RandomForestFactory, GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory]] # NOTE: run UncertainRFF after RF parameters have been obtained

experiment_iterator = product(datasets, representations, protocol_factories, method_factories)

def run_experiments(dataset, representation, protocol_factory, factory_key, dim=None, optimize=True, mock=False):
    # for dataset, representation, protocol_factory, factory_key in experiment_iterator:
    protocol_factory = protocol_factory(dataset) if type(protocol_factory) != list else protocol_factory
    for protocol in protocol_factory:
        run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                            protocol=protocol, augmentation=None, dim=dim, dim_reduction=LINEAR, mock=mock, optimize=optimize)

dim_reduction_experiment_iterator = product(["UBQT", "CALM", "1FQG"],
                                            [ONE_HOT, TRANSFORMER, EVE, ESM],
                                            [PositionalSplitterFactory, RandomSplitterFactory],
                                            method_factories,
                                            [LINEAR],
                                            [2, 10, 100, 1000])
def run_dim_reduction_experiments(dataset, representation, protocol_factory, factory_key, dim_reduction, dim, mock=False):
    for protocol in protocol_factory(dataset):
        if representation == VAE and dim and int(dim) > 30:
            continue # skip, dimensions greater than original -> None
        if representation == EVE and dim and int(dim) > 50:
            continue
        print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: {dim} {dim_reduction}, aug: None")
        run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                        protocol=protocol, augmentation=None, dim=dim, dim_reduction=dim_reduction, mock=mock)


augmentation_experiment_iterator = product(["CALM", "UBQT", "1FQG"],
                                    [EVE, TRANSFORMER, ONE_HOT, ESM],
                                    [PositionalSplitterFactory, RandomSplitterFactory],
                                    method_factories,
                                    [ROSETTA, EVE_DENSITY])
def run_augmentation_experiments(dataset, representation, protocol_factory, factory_key, augmentation, mock=False):
    for protocol in protocol_factory(dataset):
        print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()} , dim: full, aug: {augmentation}")
        run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                    protocol=protocol, augmentation=augmentation, dim=None, dim_reduction=LINEAR, mock=mock)


def run_threshold_experiments():
    for t, dataset in zip([0., 0.], ["1FQG", "UBQT"]):
        for representation in [ESM, TRANSFORMER, ONE_HOT, EVE]:
            for protocol_factory in [FractionalSplitterFactory]: # [RandomSplitterFactory, PositionalSplitterFactory] #[BioSplitterFactory("TOXI", 2, 2), BioSplitterFactory("TOXI", 2, 3), BioSplitterFactory("TOXI", 3, 3), BioSplitterFactory("TOXI", 3, 4)]:
                for protocol in protocol_factory(dataset):
                    for factory_key in [get_key_for_factory(f) for f in [GPSEFactory, GPLinearFactory, GPMaternFactory, UncertainRFFactory]]:
                        print(f"{dataset}: {representation} - {factory_key} | {protocol.get_name()}")
                        run_single_regression_task(dataset=dataset, representation=representation, method_key=factory_key,
                                                    protocol=protocol, augmentation=None, dim=None, threshold=t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Specifications")
    parser.add_argument("-d", "--data", type=str, default=datasets, choices=datasets, help="Dataset identifier")
    parser.add_argument("-r", "--representation", type=str, default=representations, choices=representations, help="Representation of data identifier")
    parser.add_argument("-p", "--protocol", type=int, default=list(range(len(protocol_factories))), help="Index for Protocol from list [Random, Positional, Fractional]")
    parser.add_argument("-m", "--method_key", type=str, default=method_factories, choices=method_factories, help="Method identifier")
    parser.add_argument("--dim", type=int, default=None, help="Dimension reduction experiments")
    parser.add_argument("--ablation", type=str, default=None, choices=["dim-reduction", "augmentation", "threshold", "cv"], help="Specify type of ablation for the run.")
    parser.add_argument("--no_optimize", action='store_true', help="Do not optimize regressor.")
    parser.add_argument("--mock", action='store_true', help="Mock experiment iterations.")
    args = parser.parse_args()
    optimize_flag = True if not args.no_optimize else False

    if "TOXI" in args.data:
        protocol_factories += biosplitter_factories
    if not args.ablation:
        # MAIN EXPERIMENT
        if not any([isinstance(param,list) for param in [args.data, args.representation, args.protocol, args.method_key]]): # if parameters are passed correctly:
            run_experiments(dataset=args.data, representation=args.representation, protocol_factory=protocol_factories[args.protocol], factory_key=args.method_key, dim=args.dim, optimize=optimize_flag, mock=args.mock)
        else: # per default iterate over everything
            data = args.data if isinstance(args.data, list) else [args.data]
            rep = args.representation if isinstance(args.representation, list) else [args.representation]
            protocol_idx = args.protocol if isinstance(args.protocol, list) else [args.protocol]
            method = args.method_key if isinstance(args.method_key, list) else [args.method_key]
            param_iterator = product(data, rep, protocol_idx, method)
            for d, r, p_idx, m in param_iterator:
                run_experiments(dataset=d, representation=r, protocol_factory=protocol_factories[p_idx], factory_key=m, dim=args.dim, optimize=optimize_flag, mock=args.mock)
    # ABLATION STUDIES: (dim-reduction, augmentation, threshold)
    elif args.ablation == "dim_reduction":
        for dataset, representation, protocol_factory, factory_key, dim_reduction, dim in dim_reduction_experiment_iterator:
            run_dim_reduction_experiments(dataset, representation, protocol_factory, factory_key, dim_reduction, dim) 
    elif args.ablation == "augmentation":
        for dataset, representation, protocol_factory, factory_key, augmentation in augmentation_experiment_iterator:
            run_augmentation_experiments(dataset, representation, protocol_factory, factory_key, augmentation) 
    elif args.ablation == "threshold":
        run_threshold_experiments() # TODO: parallelize
    elif args.ablation == "cv":
        ablate_methods = [get_key_for_factory(f) for f in [KNNFactory, GPLinearFactory, GPMaternFactory]]
        ablation_protocols_random_cv_iterator = product(["TIMB"],
                                            [ONE_HOT, TRANSFORMER, EVE, ESM],
                                            ablation_protocols_random_cv,
                                            ablate_methods)
        for d, r, p, m in ablation_protocols_random_cv_iterator:
            assert d == p[0].dataset, f"Mismatch between dataset {d} and instantiated protocol dataset {p[0].dataset}!"
            run_experiments(dataset=d, representation=r, protocol_factory=p, factory_key=m, mock=args.mock)
    else:
        raise NotImplementedError("The provided configuration does not exist!")
    
