#!/bin/bash

# Use esm2 + structure features
EMBEDDING_TAG="esm2_struc"              # for ckpt path & output naming
EMBEDDING_LIST=(esm2 esm_if gearnet)    # passed to --embedding_list

BASE_DIR="../CAGI7"
PYTHON_SCRIPT="inference.py"
CKPT_BASE_PATH="../ckpt/${EMBEDDING_TAG}"

# Datasets to process
CSV_FILES=("ARSA.csv" "ATP7B.csv" "LPL.csv" "TSC2.csv")

for CSV_FILE in "${CSV_FILES[@]}"; do
    echo "----------------------------------------"
    DATASET_NAME="${CSV_FILE%.csv}"
    OUTPUT_DIR="${BASE_DIR}/${DATASET_NAME}"
    PDB_PATH="${BASE_DIR}/${DATASET_NAME}_HUMAN.pdb"   # structure file per dataset

    mkdir -p "${OUTPUT_DIR}"

    echo "Processing dataset: ${DATASET_NAME}"

    # folds 0..4 (test_fold is always 0 per your setup)
    for FOLD in {0..4}; do
        echo "  - Running inference for fold ${FOLD}..."

        python "${PYTHON_SCRIPT}" \
            --embedding_list ${EMBEDDING_LIST[@]} \
            --test_fold 0 \
            --ckpt_path "${CKPT_BASE_PATH}/fold${FOLD}/mlp_best_fold${FOLD}.pt" \
            --dms_csv "${BASE_DIR}/${CSV_FILE}" \
            --pdb_path "${PDB_PATH}" \
            --output_file "${OUTPUT_DIR}/${DATASET_NAME}_${EMBEDDING_TAG}_fold${FOLD}.csv"

        echo "  - Inference for fold ${FOLD} complete."
    done

    echo "Dataset ${DATASET_NAME} processing complete."
done

echo "----------------------------------------"
echo "All datasets have been processed."

