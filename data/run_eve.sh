conda activate eve_env

/home/pcq275/unmount_erda.sh
/home/pcq275/mount_erda.sh

MNT_DIR=/home/pcq275/mnt
MSA_DIR=${MNT_DIR}
MSA_LIST=${MNT_DIR}/eve_mappings.csv
VAE_CHECKPOINT=${MNT_DIR}/eve_checkpoint
MODEL_PARAMETERS=/home/pcq275/EVE/EVE/default_model_params.json

idx=0

python /home/pcq275/EVE/train_VAE.py \
    --MSA_data_folder ${MSA_DIR} \
    --MSA_list ${MSA_LIST} \
    --protein_index ${idx} \
    --VAE_checkpoint_location ${VAE_CHECKPOINT} \
    --model_parameters_location ${MODEL_PARAMETERS} \
    --training_logs_location ${VAE_CHECKPOINT}/logs/ \
    --MSA_weights_location ${VAE_CHECKPOINT} \
    --model_name_suffix 06_22