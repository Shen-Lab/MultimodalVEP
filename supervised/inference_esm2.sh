#!/bin/bash

# Define common variables
EMBEDDING="esm2"
BASE_DIR="../CAGI7"
PYTHON_SCRIPT="inference.py"
CKPT_BASE_PATH="../ckpt/${EMBEDDING}"

# List of CSV files to process
# Add or remove files from this list as needed
CSV_FILES=("ARSA.csv" "ATP7B.csv" "LPL.csv" "TSC2.csv")

# Loop through each CSV file in the list
for CSV_FILE in "${CSV_FILES[@]}"
do
    echo "----------------------------------------"
    # Extract the base name (e.g., LPL from LPL.csv)
    # The ##* removes the longest match of a string from the beginning of a variable
    # The %%* removes the longest match of a string from the end of a variable
    DATASET_NAME="${CSV_FILE%.csv}"
    OUTPUT_DIR="${BASE_DIR}/${DATASET_NAME}"
    
    # Create the output directory for this dataset if it doesn't exist
    mkdir -p ${OUTPUT_DIR}
    
    echo "Processing dataset: ${DATASET_NAME}"
    
    # Loop through the 5 folds (from 0 to 4)
    for FOLD in {0..4}
    do
        echo "  - Running inference for fold ${FOLD}..."

        # The core command with updated variables
        python ${PYTHON_SCRIPT} \
            --embedding_list ${EMBEDDING} \
            --test_fold 0 \
            --ckpt_path ${CKPT_BASE_PATH}/fold${FOLD}/mlp_best_fold${FOLD}.pt \
            --dms_csv ${BASE_DIR}/${CSV_FILE} \
            --output_file ${OUTPUT_DIR}/${DATASET_NAME}_${EMBEDDING}_fold${FOLD}.csv

        echo "  - Inference for fold ${FOLD} complete."
    done
    
    echo "Dataset ${DATASET_NAME} processing complete."
done

echo "----------------------------------------"
echo "All datasets have been processed."
