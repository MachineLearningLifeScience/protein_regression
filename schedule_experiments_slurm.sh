#!/bin/bash
#SBATCH --job-name=PositionPR_BENCH
#SBATCH -p boomsma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --array=59
#SBATCH --time=12:00:00
#SBATCH --exclude=hendrixgpu13fl


OUTPUT_LOG=/home/pcq275/protein_regression/slurm_experiment_run_out_position.log

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate protein_regression

CONFIG=/home/pcq275/protein_regression/slurm_experiment_config_no_GP_positionsplitter.txt

dataset=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
representation=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
protocol=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})
method=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})

echo "python /home/pcq275/protein_regression/run_experiments.py -d ${dataset} -r ${representation} -p ${protocol} -m ${method}" >> ${OUTPUT_LOG}
python /home/pcq275/protein_regression/run_experiments.py -d ${dataset} -r ${representation} -p ${protocol} -m ${method} >> ${OUTPUT_LOG}
