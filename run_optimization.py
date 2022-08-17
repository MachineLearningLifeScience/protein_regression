from algorithm_factories import GPLinearFactory, get_key_for_factory, ALGORITHM_REGISTRY
from algorithm_factories import GPMaternFactory, GPSEFactory, UncertainRFFactory
import data
from run_single_optimization_task import run_single_optimization_task, run_ranked_reference_task
from util.execution.cluster import execute_job_array_on_slurm_cluster
from util.mlflow.constants import EVE_DENSITY, TRANSFORMER, ONE_HOT, VAE, ESM, VAE_DENSITY, VAE_AUX, VAE_RAND, EVE, AT_RANDOM

RUN_ON_CLUSTER = False

datasets = ["1FQG", "UBQT", "CALM"] # "UBQT", "CALM", "1FQG"
representations = [ONE_HOT] # TRANSFORMER, VAE_RAND, VAE_AUX, ONE_HOT, ESM, EVE
seeds = [11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245] # 11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245
#seeds = [11]
max_iterations = 500
run_reference_scoring = False

method_factory_keys = ALGORITHM_REGISTRY.keys()
method_factory_keys = [get_key_for_factory(f) for f in [GPMaternFactory]] # GPSEFactory, GPMaternFactory, UncertainRFFactory, GPLinearFactory
commands = []
command_template = "python run_single_optimization_task.py -d %s -s %i -r %s -m %s"
for dataset in datasets:
    if run_reference_scoring:
        run_ranked_reference_task(dataset, max_iterations=max_iterations, reference_task=EVE_DENSITY)
    for seed in seeds:
        run_ranked_reference_task(dataset, max_iterations=max_iterations, reference_task=AT_RANDOM, seed=seed)
        for representation in representations:
            for method in method_factory_keys:
                if RUN_ON_CLUSTER:
                    commands.append(command_template % (dataset, seed, representation, method))
                else:
                    run_single_optimization_task(dataset, method, seed, representation, max_iterations=max_iterations, log_interval=1)
# print(commands[0])
if RUN_ON_CLUSTER:
    execute_job_array_on_slurm_cluster(commands, cpus=8)
