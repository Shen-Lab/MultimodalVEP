#!/bin/bash

## NESSARY JOB SPECIFICATIONS
#SBATCH --job-name=clinvar_eval
#SBATCH --time=48:00:00                 # adjust if needed
#SBATCH --ntasks=1
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# environment setup
conda env list
source /home/yxliu/.bashrc
source activate mep_env
cd ../supervised/
export PYTHONUNBUFFERED=TRUE
module load WebProxy
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_HOME='/scratch/user/yxliu/MEP-SiPLM/ckpt'
export HF_HOME='/scratch/user/yxliu/MEP-SiPLM/ckpt'
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"

# datasets (clinvar CSVs)
CSV_FILES=("clinvar_LPL.csv" "clinvar_ARSA.csv" "clinvar_ATP7B.csv" "clinvar_TSC2.csv")

# model configs
# tag : embedding_list : ckpt_dir
CONFIGS=(
  "saprot_struc:saprot esm_if gearnet:../ckpt/saprot_struc"
  "esm2:esm2:../ckpt/esm2"
  "esm2_struc:esm2 esm_if gearnet:../ckpt/esm2_struc"
)

PYTHON_SCRIPT="inference.py"
BASE_DIR="../CAGI7"

for CSV_FILE in "${CSV_FILES[@]}"; do
    DATASET_NAME="${CSV_FILE%.csv}"           # e.g. clinvar_LPL
    BASE_PROT="${DATASET_NAME#clinvar_}"      # e.g. LPL
    OUTPUT_DIR="${BASE_DIR}/${BASE_PROT}"
    PDB_PATH="${BASE_DIR}/${BASE_PROT}_HUMAN.pdb"
    mkdir -p "${OUTPUT_DIR}"

    echo "Processing dataset: ${DATASET_NAME} (${BASE_PROT})"

    for config in "${CONFIGS[@]}"; do
        IFS=":" read -r TAG EMBEDDINGS CKPT_BASE <<< "${config}"
        echo "  Model type: ${TAG} with embeddings [${EMBEDDINGS}]"

        for FOLD in {0..4}; do
            echo "    Running fold ${FOLD}..."
            python "${PYTHON_SCRIPT}" \
                --embedding_list ${EMBEDDINGS} \
                --test_fold 0 \
                --ckpt_path "${CKPT_BASE}/fold${FOLD}/mlp_best_fold${FOLD}.pt" \
                --dms_csv "${BASE_DIR}/${CSV_FILE}" \
                --pdb_path "${PDB_PATH}" \
                --output_file "${OUTPUT_DIR}/${DATASET_NAME}_${TAG}_fold${FOLD}.csv"
        done
    done
done

echo "All clinvar evaluations complete."

