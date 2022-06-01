#!/usr/bin/env/ zsh 

# BASH >=4. can be used as well - default OSX bash 3.2 breaks for associative arrays!!

# create key-value pair of dataset with associated alignments
declare -A ALIGNMENTS_ARRAY=( ["CALM"]="CALM1_HUMAN_1_b0.5" ["MTH3"]="MTH3_HAEAESTABILIZED_1_b0.5" ["BLAT"]="BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105" ["UBQT"]="RL401_YEAST_1_b0.5" ["BRCA"]="BRCA1_HUMAN_1_b0.5" ["TIMB"]="TRPC_THEMA_1_b0.5" ["TOXI"]="parEparD_3")

# NOT the alignment used in BLAT_df: ["BLAT"]="BLAT_ECOLX_1_b0.5"

# check if mmseq exists
type mmseqs >/dev/null 2>&1 || { echo >&2 "MMseqs2 required but it's not installed! Exiting..."; exit 1; }

# if exists -> create required temporary directory
WORKING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="${WORKING_DIR}/files/alignments/"
TEMP_DIR="${WORKING_DIR}/tmp"

mkdir -p ${TEMP_DIR}

# check if tmp dir was created
if [[ ! "${TEMP_DIR}" || ! -d "${TEMP_DIR}" ]]; then
  echo "Could not create temp dir"
  exit 1
fi


function run_mmseq {
    for dataset filename in ${(kv)ALIGNMENTS_ARRAY[@]}; do
        alignment_filename="${WORKING_DIR}/files/alignments/${filename}.a2m"
        if [ -f "${alignment_filename}" ]; then
                echo "${dataset} alignment is found!"
            else
                echo "${dataset} is not found under ${alignment_filename}"
                exit 1
        fi
        # create mmseq DB from alignment fasta/a2m
        db=${OUTPUT_DIR}/${dataset}_DB
        output_tsv=${OUTPUT_DIR}/${dataset}_mmseqs2_si02_ac08_clusters.tsv
        output_fasta=${OUTPUT_DIR}/${dataset}_mmseqs2_si02_ac08_seqs.fasta
        mmseqs createdb ${alignment_filename} ${db}
        # perform clustering with sequence identity and alignment coverage
        mmseqs cluster ${db} ${db}_clu ${TEMP_DIR} --min-seq-id 0.2 -c 0.8
        # create TSV from output DB_clu
        mmseqs createtsv ${db} ${db} ${db}_clu ${output_tsv}
        # create sequence file after clustering
        mmseqs createseqfiledb ${db} ${db}_clu ${dataset}_clu_seq
        # flatten into fasta file
        mmseqs result2flat ${db} ${db} ${dataset}_clu_seq ${output_fasta}
        # cleanup dataset DB files
        rm ${db}* && rm ${WORKING_DIR}/${dataset}_clu*
    done
}

run_mmseq

echo "Succesfully clustered dataset alignments!"

# cleaning up tmp dir
rm -rf "${TEMP_DIR}"

exit 0
