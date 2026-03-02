# Protein Mutation Effect Prediction Using Structure Information and Protein Language Model

## Submitting Jobs to HPRC
Before proceeding, it's important to know how to utilize Grace's job scheduler (Slurm). Submitting jobs allows us for: running programs in the background, using specified hardware, automatic logging, etc. Please follow this general template when creating ```.sh``` scripts.
```
#!/bin/bash
#SBATCH --job-name=<job_name>               # Set the <job_name> to an appropriate name
#SBATCH --time=24:00:00                     # Set the wall clock limit
#SBATCH --ntasks=1                          # Request number of tasks
#SBATCH --mem=200G                          # Request memory per node
#SBATCH --output=logs/<output_name>%j.out   # Send stdout to "<output_name>.[jobID]"
#SBATCH --error=logs/<error_name>%j.err     # Send error to "<error_name>.[jobID]"
#SBATCH --gres=gpu:a100:1                   # Request number GPUs per node (1 or 2)
#SBATCH --partition=gpu                     # Request the GPU partition/queue

# for online capabilities (optional)
module load WebProxy

# activate the conda environment
source ~/.bashrc
conda activate <env_name>

# open file directory
cd /your_path/

# run python file
python <file>.py
```
To submit a job, run ```sbatch name.sh```. You can monitor your jobs via the OnDemand portal. For more info on submitting jobs, reference these resources: https://hprc.tamu.edu/kb/Quick-Start/Grace/#running-your-program-preparing-a-job-file and https://hprc.tamu.edu/files/training/2021/Fall/GracePrimer_HPRC_2021_fall.pdf

## Prerequisites:
Setting up environment
```
conda create -n mep_env python=3.9
conda activate mep_env # open the environment after creating it
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/esm.git #newest version needed for ESM-IF (https://github.com/facebookresearch/esm/pull/386)
pip install transformers
pip install scipy
pip install pandas
pip install h5py

# optional: only needed for GearNet
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install torchdrug
pip install easydict pyyaml

#optional: only needed for SaProt
conda install -c conda-forge -c bioconda foldseek

#optional: only needed for ESM-IF
pip install torch_geometric
pip install biotite==0.41.1
```
## Installation
Clone the repository
```
cd $SCRATCH
git clone --branch base https://github.com/Shen-Lab/MultimodalVEP.git
```
Download the ProteinGym dataset. A slurm script is already provided for you. You just need to modify the file path. Please download BOTH ckpt.zip and dataset.zip from (https://zenodo.org/records/10976493).
```
cd MultimodalVEP/jobs # path to the slurm script

sbatch download.sh
```

## File tree
```
--benchmark: scripts to evaluate the previous models performance\
--data: scripts to preprocess the dataset\
--dataset: ProteinGym dataset and related files. (unzip from Zenodo link)
```

## Zero-shot Mutation Effect Inference
```
cd benchmark
python ESM2.py # ESM2 zero-shot mutation effect prediction using wild-type marginal and masked marginal methods: https://huggingface.co/blog/AmelieSchreiber/mutation-scoring
```
## Precomputer Embedding

Not needed if you download the dataset from Zenodo in the Installation Step
File list can be found: https://drive.google.com/drive/folders/1xB43lm6M-MuwqP4KLqEruJ4GBIuLURQY?usp=sharing
```
cd data
python get_esm_embedding.py --file_list job_12h_1_files.txt [choose from 1 to 4, each will spend around 15h using A100]
python get_SaProt_embedding.py --file_list job_12h_1_files.txt [choose from 1 to 4]
python get_GearNet_embedding.py
python get_esm_IF_embedding.py
```
## Model Training
```
cd supervised
python train.py --embedding_list esm2 esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/esm2_struc/fold0/ #embedding list choose from: [saprot, esm2, esm_if, gearnet], test_fold: [0,1,2,3,4]
```
## Evaluation
```
python evaluation.py  --embedding_list esm2 esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/esm2_struc/fold0/mlp_best_fold0.pt --dms_csv ../dataset/ProteinGym/substitution_split/A0A1I9GEU1_NEIME_Kennouche_2019.csv #make sure embedding list is the same as your training list
```
## To Do:
```
1. 7 proteins' sequence and structure not match: seq_id: {A0A140D2T1_ZIKV_Sourisseau_2019, BRCA2_HUMAN_Erwood_2022_HEK293T, CAS9_STRP1_Spencer_2017_positive, P53_HUMAN_Giacomelli_2018_Null_Etoposide, P53_HUMAN_Giacomelli_2018_Null_Nutlin, P53_HUMAN_Giacomelli_2018_WT_Nutlin,
POLG_HCVJF_Qi_2014,}. skipped them for now.

2. Several sequence have multi-mutation sequences. Skipped them for now.
```
