from algorithm_factories import GPLinearFactory, get_key_for_factory, ALGORITHM_REGISTRY, GPMaternFactory, GPSEFactory, UncertainRFFactory
from run_single_optimization_task import run_single_optimization_task
from util.execution.cluster import execute_job_array_on_slurm_cluster
from util.mlflow.constants import TRANSFORMER, ONE_HOT, VAE, ESM

RUN_ON_CLUSTER = False

datasets = ["1FQG"] # "UBQT", "CALM", "1FQG"
representations = [TRANSFORMER] # TRANSFORMER, VAE, ONE_HOT, ESM
seeds = [11, 42, 123, 54, 2345, 987, 6538, 78543, 3465, 43245] # 11, 42, 123, 54, 2345, 987
#seeds = [11]


method_factory_keys = ALGORITHM_REGISTRY.keys()
method_factory_keys = [get_key_for_factory(f) for f in [GPSEFactory, UncertainRFFactory]] # TODO: GPLinearFactory
commands = []
command_template = "python run_single_optimization_task.py -d %s -s %i -r %s -m %s"
for dataset in datasets:
    for seed in seeds:
        for representation in representations:
            for method in method_factory_keys:
                if RUN_ON_CLUSTER:
                    commands.append(command_template % (dataset, seed, representation, method))
                else:
                    run_single_optimization_task(dataset, method, seed, representation, max_iterations=500)
# print(commands[0])
if RUN_ON_CLUSTER:
    execute_job_array_on_slurm_cluster(commands, cpus=8)
