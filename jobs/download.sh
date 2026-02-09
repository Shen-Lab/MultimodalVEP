#!/bin/bash
#SBATCH --job-name=zenodo                   # Set the <job_name> to an appropriate name
#SBATCH --time=24:00:00                     # Set the wall clock limit
#SBATCH --ntasks=1                          # Request number of tasks
#SBATCH --mem=200G                          # Request memory per node
#SBATCH --output=logs/zenodo%j.out          # Send stdout to "<output_name>.[jobID]"
#SBATCH --error=logs/zenodo%j.err           # Send error to "<error_name>.[jobID]"
#SBATCH --gres=gpu:a100:1                   # Request number GPUs per node (1 or 2)
#SBATCH --partition=gpu                     # Request the GPU partition/queue

module load WebProxy
module load wget 2>/dev/null || true

cd /file_path/

wget -O dataset.zip "https://zenodo.org/records/10976493/files/dataset.zip?download=1"
wget -O ckpt.zip "https://zenodo.org/records/10976493/files/ckpt.zip?download=1"

unzip -o dataset.zip
unzip -o ckpt.zip