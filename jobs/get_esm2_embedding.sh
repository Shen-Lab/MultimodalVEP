#!/bin/bash

##NESSARY JOB SPECIFICATIONS
#SBATCH --job-name=deformation_flowmatching       #Set the job name to "JobExample4"
#SBATCH --time=20:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=180G                  #Request 2560MB (2.5GB) per node
#SBATCH --gres=gpu:a100:1                #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --array=1-4

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132715540063             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=yxliu@tamu.edu    #Send all emails to email_address 

#First Executable Line
conda env list
source /home/yxliu/.bashrc
source activate mep_env
cd ../data/
export PYTHONUNBUFFERED=TRUE
module load WebProxy
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

FILE_LIST="job_12h_${SLURM_ARRAY_TASK_ID}_files.txt"
echo "This task will process files listed in: ${FILE_LIST}"

python get_esm_embedding.py --file_list ${FILE_LIST}

