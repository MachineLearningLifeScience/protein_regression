import argparse
from typing import List

import data
from algorithm_factories import (ALGORITHM_REGISTRY, GPLinearFactory,
                                 GPMaternFactory, GPSEFactory,
                                 UncertainRFFactory, get_key_for_factory)
from run_single_optimization_task import (run_ranked_reference_task,
                                          run_single_optimization_task)
from util.mlflow.constants import (AT_RANDOM, ESM, EVE, EVE_DENSITY, ONE_HOT,
                                   TRANSFORMER)

RUN_ON_CLUSTER = False

datasets = ["UBQT", "CALM", "1FQG"]
representations = [TRANSFORMER, ONE_HOT, ESM, EVE]
seeds = [11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245] # 11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245
max_iterations = 500

method_factory_keys = ALGORITHM_REGISTRY.keys()
method_factory_keys = [get_key_for_factory(f) for f in [GPSEFactory, GPMaternFactory, UncertainRFFactory, GPLinearFactory]] # GPSEFactory, GPMaternFactory, UncertainRFFactory, GPLinearFactory

def optimization_experiment(datasets: List[str], method_keys: List[str], 
                        representations: List[str], seeds: List[int], 
                        budget: int=max_iterations, reference_scoring: bool=True) -> None:
    for dataset in datasets:
        if reference_scoring:
            run_ranked_reference_task(dataset, max_iterations=max_iterations, reference_task=EVE_DENSITY)
        for seed in seeds:
            run_ranked_reference_task(dataset, max_iterations=max_iterations, reference_task=AT_RANDOM, seed=seed)
            for representation in representations:
                for method in method_keys:
                    run_single_optimization_task(dataset, method, seed, representation, max_iterations=budget, log_interval=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Specifications")
    parser.add_argument("-d", "--data", type=str, default=datasets, choices=datasets, help="Dataset identifier.")
    parser.add_argument("-r", "--representation", type=str, default=representations, choices=representations, help="Representation of data identifier.")
    parser.add_argument("-m", "--method_key", type=str, default=method_factory_keys, choices=method_factory_keys, help="Method identifier.")
    parser.add_argument("-s", "--seeds", type=int, default=seeds, help="Random seed.")
    parser.add_argument("-b", "--budget", type=int, default=max_iterations, help="Number of optimization iterations.")
    args = parser.parse_args()
    datasets = [args.data] if not isinstance(args.data, list) else args.data
    methods = [args.method_key] if not isinstance(args.method_key, list) else args.method_key
    representations = [args.representation] if not isinstance(args.representation, list) else args.representation
    seeds = [args.seeds] if not isinstance(args.seeds, list) else args.seeds

    optimization_experiment(datasets=datasets, method_keys=methods, representations=representations, 
                budget=args.budget, seeds=seeds)
