import os
from os.path import sep
from time import time
import pickle


def execute_single_configuration_on_slurm_cluster(command: str, cpus: int):
    file_name = "./cluster_run_scripts/" + command.replace(' ', '') + ".sq"
    with open(file_name, "w+") as fh:
        _write_basic_run_script(fh, cpus)
        fh.writelines(command)
        fh.flush()
        cluster_command = "sbatch --exclusive --ntasks=1 --mem=30000M --time=0-02:00:00 --cpus-per-task=%i %s" % (cpus, file_name)
        os.system(cluster_command)
        return cluster_command


def execute_job_array_on_slurm_cluster(commands: [str], cpus: int):
    stamp = str(time())
    f = open(os.getcwd() + sep + "cluster_run_scripts" + sep + stamp + ".pkl", "wb+")
    pickle.dump(commands, f)
    f.flush()
    f.close()

    file_name = "./cluster_run_scripts/" + stamp + ".sq"
    with open(file_name, "w+") as fh:
        _write_basic_run_script(fh, cpus)
        fh.writelines("python run_single_array_configuration.py -t %s -j $SLURM_ARRAY_TASK_ID" % stamp)
        fh.flush()
        cluster_command = "sbatch --exclusive --ntasks=1 --mem=30000M --time=0-02:00:00 --cpus-per-task=%i --array=0-%i%%5 %s" % (cpus, len(commands)-1, file_name)
        os.system(cluster_command)
        return cluster_command


def _write_basic_run_script(file_handle, cpus):
    file_handle.writelines("#!/bin/bash\n")
    file_handle.writelines("CONDA_BASE=$(conda info --base) \n")
    file_handle.writelines("source $CONDA_BASE/etc/profile.d/conda.sh \n")
    file_handle.writelines("conda activate ./env \n")
    file_handle.writelines("taskset --cpu-list %s " % str([i for i in range(0, cpus)])[1:-1].replace(' ', ''))  # NO \n!