#!/bin/bash

# This script evaluates different protein language models on ClinVar datasets.
# It is a direct translation of the SLURM script, removing job scheduler commands.

# --- Environment Setup ---
# Ensure the necessary conda environment is activated before running this script.
# For example, you can run: `source activate mep_env`
echo "Setting up environment..."
export PYTHONUNBUFFERED=TRUE
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_HOME='./ckpt'
export HF_HOME='./ckpt'
echo "Environment setup complete."
echo "Note: This script assumes you are running from the parent directory of 'supervised/'"

# --- Configuration ---
PYTHON_SCRIPT="inference.py"
BASE_DIR="../CAGI7"

# Datasets (ClinVar CSVs)
CSV_FILES=(
  "clinvar_LPL.csv"
  "clinvar_ARSA.csv"
  "clinvar_ATP7B.csv"
  "clinvar_TSC2.csv"
)

# Model configurations: tag : embedding_list : ckpt_dir
CONFIGS=(
  "saprot_struc:saprot esm_if gearnet:../ckpt/saprot_struc"
  "esm2:esm2:../ckpt/esm2"
  "esm2_struc:esm2 esm_if gearnet:../ckpt/esm2_struc"
)

# --- Main Logic ---
for CSV_FILE in "${CSV_FILES[@]}"; do
    DATASET_NAME="${CSV_FILE%.csv}"
    BASE_PROT="${DATASET_NAME#clinvar_}"
    OUTPUT_DIR="${BASE_DIR}/${BASE_PROT}"
    PDB_PATH="${BASE_DIR}/${BASE_PROT}_HUMAN.pdb"
    mkdir -p "${OUTPUT_DIR}"

    echo "--- Processing dataset: ${DATASET_NAME} (${BASE_PROT}) ---"

    for config in "${CONFIGS[@]}"; do
        IFS=":" read -r TAG EMBEDDINGS CKPT_BASE <<< "${config}"
        echo "  - Model type: ${TAG} with embeddings [${EMBEDDINGS}]"

        for FOLD in {0..4}; do
            echo "    - Running fold ${FOLD}..."
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

echo "--- All ClinVar evaluations complete. ---"
