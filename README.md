# Protein Regression Assessment


## Installation

Use anaconda to install the required virtual environment from the listed ``environment_*.yml`` files,

``conda create --name protein_regression --file environment_*.yml``
(replace the asterisk with the correct system specifications).
Note that the ``environment_nix_SLURM.yml`` also applies to common Linux distributions.

### Alternatives
You are free to set up your own environment.
Please note that in order to run the experiments the project environment should contain at least the following libraries (in non-conflicting version specifications):
- numpy
- scipy
- tensorflow
- tensorflow-probability
- gpflow
- scikit-learn
- mlflow

and either NVIDIA-CUDA support or MacOS (M1) ``metal`` support.


## Replicating results
After installing the ``protein_regression`` enviroment and activating it,
please download the required data into the ``./data`` directory.

All data required to replicate the experiments can be found as an archived directory: ``https://erda.ku.dk/archives/31bf8a65612c3d2c76167143f65cd86e/published-archive.html``.

To run the experiments in the default configuration: ``python run_experiments.py``.

To run the optimization requirements: ``python run_optimization.py``.


### Running specific settings and experiments

To run specific experiments settings provide the specifications as input flags to the experiment scripts.
For example, we want to run the Beta-Lactamase experiments using an esm-1b embedding with a linear GP regressor using a RandomCV protocol:
``python run_experiments.py -d 1FQG -r esm -m GPLinearFactory -p 0``.

See the ``python run_experiments.py --help`` for more details.

## Project Structure

- ``./algorithms/`` contains abstract and implementation of the regressors,
- ``./data/`` contains scripts to compute embeddings/representations, and splitting protocols,
- ``./data/files`` contains the required data-sets to run experiments, which includes (original .csv files, embeddings, MSA files), the persisted files are in pickle format,
- ``./notebooks/`` contains jupyter notebooks to replicate the figures from the manuscript; requires that experiments have run and completed succesfully,
- ``./notebooks/figures_main.ipynb`` contains the figures for the main manuscript,
- ``./notebooks/figures_supplementary.ipynb`` contains the figures for the supplementary material, 
- ``./results/`` directory for experimental results, ``./results/cache/`` caching of dictionaries obtained from MlFlow, ``./results/figures/`` saved figures obtained from ``./make_plot_*.py`` scripts, ``./results/mlruns`` output of MlFlow experiments,
- ``./test/`` contains pytest modules for specific tests, i.e. data-loading and consistency, tests of custom CV splitters, custom UC/UQ code,
- ``./uncertainty_quantification/`` module for UC/UQ code
- ``./util/`` miscalleanous utility code, used for encoding of data, pre-, and post-processing
- ``./util/mlflow/`` MlFlow specific utility module, defines variables, constants, loading functions, etc.
- ``./visualization/`` module required to generate figures; required by ``./make_plot_*.py`` scripts,
- ``./run_experiments.py`` run script for all experiments; calls ``./run_single_regression_task.py`` with experiments specifications,
- ``./run_optimization.py`` run script for all experiments with the optimization protocol; calls ``./run_single_optimization_task.py`` with experiment specifications,
- ``./schedule_experiments_slurm*.sh`` shell scripts to schedule slurm runs as assay jobs, requires files under ``./slurm_configs/`` as experiment input parameters,

