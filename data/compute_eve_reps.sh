#!/bin/bash

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate protein_env

#/home/pcq275/unmount_erda.sh
#/home/pcq275/mount_erda.sh

MNT_DIR=/home/pcq275/mnt_pr
MSA_DIR=${MNT_DIR}/alignments
MSA_LIST=${MNT_DIR}/eve_mappings.csv
VAE_CHECKPOINT=${MNT_DIR}/eve_checkpoint
MODEL_PARAMETERS=/home/pcq275/EVE/EVE/default_model_params.json
SUFFIX=06_22

computation_mode='all_singles'
all_singles_mutations_folder=${MNT_DIR}/protein_gym/substitutions_raw_DMS/
output_evol_indices_location=${MNT_DIR}/eve_results/ # output_evol_indices_location=${MNT_DIR}/eve_results_dec/
num_samples_compute_evol_indices=2000
batch_size=2048

for idx in $(seq 0 6); do
# BLAT is idx=1
echo ${idx}
python /home/pcq275/EVE/compute_evol_indices.py \
   --MSA_data_folder ${MSA_DIR} \
   --MSA_list ${MSA_LIST} \
   --protein_index ${idx} \
   --MSA_weights_location ${VAE_CHECKPOINT} \
   --VAE_checkpoint_location ${VAE_CHECKPOINT} \
   --model_name_suffix ${SUFFIX} \
   --model_parameters_location ${MODEL_PARAMETERS} \
   --computation_mode ${computation_mode} \
   --all_singles_mutations_folder ${all_singles_mutations_folder} \
   --output_evol_indices_location ${output_evol_indices_location} \
   --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
   --batch_size ${batch_size}
done

# !WARN! SPECIAL CASE: BRCA1 , TOXI
# FOR BRCA all_singles list != the experimental mutations:
idx=7 # idx=4 BRCA1; idx=7 for TOXI
computation_mode='input_mutations_list'
## MUTATIONS: 	BRCA => BRCA1_HUMAN_Findaly_2018.csv
##		TOXI => TOXI.csv
mutations_location=/home/pcq275/EVE/data/mutations # -> BRCA.csv == BRCA1_HUMAN_Findlay_2018.csv
python /home/pcq275/EVE/compute_evol_indices.py \
    --MSA_data_folder ${MSA_DIR} \
    --MSA_list ${MSA_LIST} \
    --protein_index ${idx} \
    --MSA_weights_location ${VAE_CHECKPOINT} \
    --VAE_checkpoint_location ${VAE_CHECKPOINT} \
    --model_name_suffix ${SUFFIX} \
    --model_parameters_location ${MODEL_PARAMETERS} \
    --computation_mode ${computation_mode} \
    --mutations_location ${mutations_location} \
    --output_evol_indices_location ${output_evol_indices_location} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size}
