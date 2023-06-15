#!/bin/bash
#SBATCH --job-name=OptimizePR_BENCH
#SBATCH -p boomsma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --array=224-4339
#SBATCH --time=12:00:00

OUTPUT_LOG=/home/pcq275/protein_regression/slurm_experiment_optimize.log

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate protein_regression

## ENABLE CUDA ON Cluster
if [ $(cat $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh | wc -l) == 0 ]; then
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
fi

CONFIG=/home/pcq275/protein_regression/slurm_configs/slurm_experiment_optimization_config.txt

dataset=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
representation=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
seed=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})
method=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})

echo "python /home/pcq275/protein_regression/run_optimization.py -d ${dataset} -r ${representation} -s ${seed} -m ${method}" >> ${OUTPUT_LOG}
python /home/pcq275/protein_regression/run_optimization.py -d ${dataset} -r ${representation} -s ${seed} -m ${method} >> ${OUTPUT_LOG}
