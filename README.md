# Protein Regression Assessment

[![Article](https://img.shields.io/badge/article-Comp.Bio-blue)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012061)
[![Data](https://img.shields.io/badge/data-archive-darkgreen)](https://erda.ku.dk/archives/9a379e8618a1ba1f2730ec33fa3a736d/published-archive.html)
[![License](https://img.shields.io/badge/License-MIT-purple)](https://github.com/MachineLearningLifeScience/protein_regression/tree/master?tab=MIT-1-ov-file)
[![Python](https://img.shields.io/badge/python-3.9-green)](https://github.com/MachineLearningLifeScience/protein_regression/blob/master/environment_nix_SLURM.yml)

![Fig1](https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1012061.g001)

![Fig2](https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1012061.g002)

Repository to replicate the results of [A systematic analysis of regression models for protein engineering](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012061).


## Data \& Results

All data and persisted results can be found in the [Electronic Research Data Archive](https://erda.ku.dk/archives/9a379e8618a1ba1f2730ec33fa3a736d/published-archive.html) (ID=archive-xENMse).


## Installation

Use `conda` to install the required virtual environment from the listed ``environment_*.yml`` files,

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

and either `cuda` or MacOS (M1) ``metal`` support.


## Reproducing Figures
If all experiments have been completed succesfully
or if all persisted results have been downloaded either:
- run respective ``./notebooks/``
- run ``./make_plot_*`` to create figures.

## Replicating results
After installing the ``protein_regression`` environment and activating it download the required data into the ``./data`` directory.

### Files
Download
1. `{blat|brca|calm|mth3|timb|toxi|ubqt}_data_df.pkl` (sequences and observations as DataFrames)
2. `{blat|brca|calm|mth3|timb|toxi|ubqt}_{esm|esm1v|esm2|prott5|pssm|seq_reps}_rep.{pkl|npz}` (most embeddings as pickle or numpy persisted files)
3. `ProtBert_{blat|brca|calm|mth3|timb|toxi|ubqt}_labelled_seqs.pkl` (ProtBert embeddings pickled)
4. `EVE_{BLAT|BRCA|CALM|MTH3|TIMB|TOXI|UBQT}_2000_samples.csv` (EVE embeddings as csv).

To run experiments in the default configuration: ``(protein-regression) python run_experiments.py``.

To run the optimization experiments: ``(protein-regression) python run_optimization.py``.


### Running specific settings and experiments

To run __specific experiments__ settings provide the specifications as input flags to the experiment scripts.
For example, we want to run the Beta-Lactamase experiments using an esm-1b embedding with a linear GP regressor using a RandomCV protocol:
``python run_experiments.py -d 1FQG -r esm -m GPLinearFactory -p 0``.

See the ``python run_experiments.py --help`` for more details:

```
usage: run_experiments.py [-h] [-d {MTH3,TIMB,CALM,1FQG,BRCA,TOXI,UBQT}] [-r {transformer,esm,eve,eve_density,one_hot,esm1v,esm2,prott5,pssm}]
                          [-p PROTOCOL] [-m {KNNFactory,RandomForestFactory,GPSEFactory,GPLinearFactory,GPMaternFactory,UncertainRFFactory}] [--dim DIM]
                          [--ablation {dim-reduction,augmentation,threshold,cv}] [--no_optimize] [--mock]

Experiment Specifications

optional arguments:
  -h, --help            show this help message and exit
  -d {MTH3,TIMB,CALM,1FQG,BRCA,TOXI,UBQT}, --data {MTH3,TIMB,CALM,1FQG,BRCA,TOXI,UBQT}
                        Dataset identifier
  -r {transformer,esm,eve,eve_density,one_hot,esm1v,esm2,prott5,pssm}, --representation {transformer,esm,eve,eve_density,one_hot,esm1v,esm2,prott5,pssm}
                        Representation of data identifier
  -p PROTOCOL, --protocol PROTOCOL
                        Index for Protocol from list [Random, Positional, Fractional]
  -m {KNNFactory,RandomForestFactory,GPSEFactory,GPLinearFactory,GPMaternFactory,UncertainRFFactory}, --method_key {KNNFactory,RandomForestFactory,GPSEFactory,GPLinearFactory,GPMaternFactory,UncertainRFFactory}
                        Method identifier
  --dim DIM             Dimension reduction experiments
  --ablation {dim-reduction,augmentation,threshold,cv}
                        Specify type of ablation for the run.
  --no_optimize         Do not optimize regressor.
  --mock                Mock experiment iterations.
```

## Project Structure

- ``./algorithms/`` contains abstract and implementation of the regressors,
- ``./data/`` contains scripts to compute embeddings/representations, and splitting protocols,
- ``./data/files`` contains the required data-sets to run experiments, which includes (original .csv files, embeddings, MSA files), the persisted files are in pickle format - __all downloaded files go here__ (!),
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



### Cite
This codebase is element of 

Michael R, Kæstel-Hansen J, Mørch Groth P, Bartels S, Salomon J, Tian P, Hatzakis NS, Boomsma W. __A systematic analysis of regression models for protein engineering__. _PLoS Comput Biol._ 2024 May 3;20(5):e1012061. doi: 10.1371/journal.pcbi.1012061. PMID: 38701099; PMCID: PMC11095727.

__NOTE:__ if you cite us and use results or data, make sure to also cite the respective sources as indicated in the _Methods_ and _Supplementary Files_. 

```
@article{MichaelKaestel2024Systematic,
  title={A systematic analysis of regression models for protein engineering},
  author={Michael, Richard and K{\ae}stel-Hansen, Jacob and M{\o}rch Groth, Peter and Bartels, Simon and Salomon, Jesper and Tian, Pengfei and Hatzakis, Nikos S and Boomsma, Wouter},
  journal={PLOS Computational Biology},
  volume={20},
  number={5},
  pages={e1012061},
  year={2024},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

