#!/bin/bash
#SBATCH --job-name=PROT_REG_BENCH
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-10%3
#SBATCH --time=12-12:00:00

module load cuda/11.8
module load cudnn/8.6.0

OUTPUT_LOG=/home/pcq275/protein_regression/slurm_experiment_run_out.log

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate protein_regression

CONFIG=/home/pcq275/protein_regression/slurm_experiment_config.txt

dataset=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
representation=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
protocol=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})
method=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})

echo "python /home/pcq275/protein_regression/run_experiments.py -d ${dataset} -r ${representation} -p ${protocol} -m ${method}" >> ${OUTPUT_LOG}
python /home/pcq275/protein_regression/run_experiments.py ${SLURM_ARRAY_TASK_ID} >> ${OUTPUT_LOG}
