#!/bin/bash

#SBATCH --job-name=saprot_eval

#SBATCH --time=24:00:00

#SBATCH --ntasks=1

#SBATCH --mem=32G

#SBATCH --output=logs/saprot_eval_%j.out

#SBATCH --error=logs/saprot_eval_%j.err

#SBATCH --gres=gpu:a100:1

#SBATCH --partition=gpu

#SBATCH --account=132666046416



module load WebProxy

source ~/.bashrc

conda activate mep

mkdir -p ../eval_results/saprot



echo "Starting SaProt evaluation across all folds..."



for FOLD in 3; do

    echo "Evaluating SaProt fold ${FOLD}..."

    CKPT_FILE="../ckpt_results/saprot/fold${FOLD}/mlp_best_fold${FOLD}.pt"

    if [ -f "$CKPT_FILE" ]; then

        for CSV_FILE in ../dataset/ProteinGym/substitution_split/*.csv; do

            PROTEIN_NAME=$(basename "$CSV_FILE" .csv)

            echo "  Evaluating ${PROTEIN_NAME}..."

            python evaluation.py --embedding_list saprot --test_fold ${FOLD} --ckpt_path ${CKPT_FILE} --dms_csv ${CSV_FILE} >> ../eval_results/saprot/fold${FOLD}_results.txt 2>&1

        done

    else

        echo "Checkpoint not found: ${CKPT_FILE}"

    fi

done



echo "SaProt evaluation complete!"