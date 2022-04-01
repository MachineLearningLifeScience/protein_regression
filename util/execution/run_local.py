import subprocess


cmd = "CONDA_BASE=$(conda info --base);"
cmd += "source $CONDA_BASE/etc/profile.d/conda.sh;"
cmd += "conda activate ./env"


def run_local(command):
    proc = subprocess.Popen(["/bin/bash", "-c", cmd + ";" + command])
    proc.wait()
