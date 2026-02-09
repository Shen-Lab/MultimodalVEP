#write a pytorch dataset for the ProteinGym dataset
from torch.utils.data import Dataset
import pandas as pd
import os
import h5py
import torch
import torch.nn as nn

class ProteinGymDataset_backup(Dataset):
    def __init__(self, data_path,embedding_list, split_method="random", test_fold=0,ignore_files=[],split="train"):
        self.split = split
        self.data_path = data_path
        #load all csv files in the data path
        self.csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f not in ignore_files]
        # create 5 fold cross validation
        self.train_data = []
        self.test_data = []
        df_reference = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
        for f in self.csv_file:
            mutations = pd.read_csv(os.path.join(data_path, f))
            if "esm_if" in embedding_list or "gearnet" in embedding_list:
                pdb_file = df_reference[df_reference["DMS_filename"]==f]["pdb_file"].value
                

            sequences = mutations['mutated_sequence'].values
            labels = mutations['DMS_score'].values
            mutation_sites = mutations['mutant'].values
            if split_method == "random":
                folds = mutations["fold_random_5"].values
            elif split_method == "modulo":
                folds = mutations["fold_modulo_5"].values
            elif split_method == "contiguous":
                folds = mutations["fold_contiguous_5"].values
            else:
                raise ValueError("split_method must be one of 'random', 'modulo', 'contiguous'")
            #split mutations into 5 folds by fold number
            for i in range(len(folds)):
                if folds[i] == test_fold:
                    self.test_data.append((sequences[i], labels[i], mutation_sites[i],self.tokenize(structure_file,mutation_sites[i]),f))
                else:
                    self.train_data.append((sequences[i], labels[i], mutation_sites[i],self.tokenize(structure_file,mutation_sites[i]),f))

    def __len__(self):
        return len(self.train_data)
    
    def tokenize(self,structure_path,mutation_sites):
        #tokenize the sequence into a list of integers
        mutation_sites = mutation_sites.split(":")
        mutation_sites = [int(site[1:-1]) for site in mutation_sites]
        parsed_seqs = get_struc_seq("bin/foldseek",structure_path)
        _,_,combined_seq = parsed_seqs
        #apply mask to the mutation sites
        combined_seq = list(combined_seq)
        for site in mutation_sites:
            combined_seq[2*site+1] = "#"
        return self.tokenizer.tokenize(combined_seq)
    
    def __getitem__(self, idx):
        if self.split == "train":
            return self.train_data[idx]
        else:
            return self.test_data[idx]

class ProteinGymDataset_v0(Dataset):
    def __init__(self, data_path, embedding_path, embedding_list, split="train", split_method="random", test_fold=0, ignore_files=[]):
        """
        PyTorch Dataset for loading pre-computed protein embeddings.

        Args:
            data_path (str): Path to the directory containing mutant CSV files.
            embedding_path (str): Base path to the directory containing embedding folders (e.g., 'esm2', 'gearnet').
            embedding_list (list): A list of embedding types to use, e.g., ['esm2', 'gearnet'].
            split (str): 'train' or 'test'.
            split_method (str): 'random', 'modulo', or 'contiguous'.
            test_fold (int): Which fold to use for the test set (0-4).
            ignore_files (list): List of CSV filenames to ignore.
        """
        self.split = split
        self.embedding_path = embedding_path
        self.embedding_list = embedding_list

        # --- 1. Create a mapping from DMS ID to PDB ID ---
        print("Creating DMS to PDB mapping...")
        ref_df = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
        self.dms_to_pdb_map = pd.Series(
            ref_df["pdb_file"].values, 
            index=ref_df["DMS_id"]
        ).to_dict()

        # --- 2. Pre-process all CSVs to gather sample metadata ---
        print("Gathering and splitting samples...")
        all_samples = []
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f not in ignore_files]
        print(len(csv_files))

        for filename in csv_files:
            dms_id = filename.replace(".csv", "")
            mutations_df = pd.read_csv(os.path.join(data_path, filename))
            
            fold_column = f"fold_{split_method}_5"
            if fold_column not in mutations_df.columns:
                raise ValueError(f"Split column '{fold_column}' not found in {filename}")

            for _, row in mutations_df.iterrows():
                sample_info = {
                    "dms_id": dms_id,
                    "mutant_id": row["mutant"],
                    "label": row["DMS_score"],
                    "fold": row[fold_column]
                }
                all_samples.append(sample_info)

        # --- 3. Split data into training and test sets ---
        if self.split == "train":
            self.samples = [s for s in all_samples if s["fold"] != test_fold]
        else: # test
            self.samples = [s for s in all_samples if s["fold"] == test_fold]
        
        print(f"Dataset initialized. Split: {self.split}. Number of samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the metadata for the requested sample
        sample_info = self.samples[idx]
        dms_id = sample_info["dms_id"]
        mutant_id = sample_info["mutant_id"]
        label = sample_info["label"]
        
        embeddings_to_concat = []

        # --- 4. Load each specified embedding ---
        for embedding_type in self.embedding_list:
            if embedding_type in ["esm2", "saprot"]:
                # These are named by DMS ID
                h5_filename = f"{dms_id}.h5"
            elif embedding_type in ["esm_if", "gearnet"]:
                # These are named by PDB file
                pdb_id = self.dms_to_pdb_map.get(dms_id)
                if not pdb_id:
                    raise ValueError(f"Could not find PDB mapping for DMS ID: {dms_id}")
                h5_filename = f"{pdb_id}.h5"
            else:
                raise ValueError(f"Unknown embedding type: {embedding_type}")
            
            h5_path = os.path.join(self.embedding_path, embedding_type, h5_filename)
            
            with h5py.File(h5_path, 'r') as hf:
                # Load the embedding array for the specific mutant
                embedding = hf[mutant_id][()]
                # Squeeze the embedding to remove the batch dimension (e.g., from (1, 1280) to (1280,))
                embeddings_to_concat.append(embedding.squeeze())

        # --- 5. Concatenate embeddings if multiple are requested ---
        if len(embeddings_to_concat) > 1:
            final_embedding = np.concatenate(embeddings_to_concat, axis=-1)
        else:
            final_embedding = embeddings_to_concat[0]
            
        return (
            torch.tensor(final_embedding, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.float32)
        )

import os
import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import pdb
class ProteinGymDataset_v1(Dataset):
    def __init__(self, data_path, embedding_path, embedding_list, split="train", split_method="random", test_fold=0, ignore_files=[]):
        """
        PyTorch Dataset that pre-loads all protein embeddings into memory for faster training.
        """
        # --- 1. Initial Setup (same as before) ---
        ref_df = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
        self.dms_to_pdb_map = pd.Series(ref_df["pdb_file"].values, index=ref_df["DMS_id"]).to_dict()

        # --- 2. Gather Sample Metadata (same as before) ---
        all_samples_metadata = []
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f not in ignore_files]

        for filename in csv_files:
            dms_id = filename.replace(".csv", "")
            mutations_df = pd.read_csv(os.path.join(data_path, filename))
            fold_column = f"fold_{split_method}_5"
            if fold_column not in mutations_df.columns:
                raise ValueError(f"Split column '{fold_column}' not found in {filename}")

            for _, row in mutations_df.iterrows():
                all_samples_metadata.append({
                    "dms_id": dms_id,
                    "mutant_id": row["mutant"],
                    "label": row["DMS_score"],
                    "fold": row[fold_column]
                })

        # --- 3. Split Metadata into Train/Test (same as before) ---
        if split == "train":
            samples_to_load = [s for s in all_samples_metadata if s["fold"] != test_fold]
        else:
            samples_to_load = [s for s in all_samples_metadata if s["fold"] == test_fold]
        
        # Group samples by DMS_id to load H5 files efficiently
        grouped_samples = defaultdict(list)
        for sample in samples_to_load:
            grouped_samples[sample["dms_id"]].append(sample)

        # --- 4. ✨ Pre-load all data into memory ---
        print(f"Pre-loading {len(samples_to_load)} samples for '{split}' split into memory...")
        self.loaded_data = []
        
        for dms_id, sample_group in tqdm(grouped_samples.items(), desc="Loading embedding files"):
            # Open all required H5 files for this dms_id just once
            h5_files = {}
            pdb.set_trace()
            for emb_type in embedding_list:
                if emb_type in ["esm2", "saprot"]:
                    h5_filename = f"{dms_id}.h5"
                else: # esm_if, gearnet
                    pdb_id = self.dms_to_pdb_map.get(dms_id)
                    if not pdb_id: pdb.set_trace()
                    pdb_id = pdb_id.split(".")[0]
                    h5_filename = f"{pdb_id}.h5"
                
                h5_path = os.path.join(embedding_path, emb_type, h5_filename)
                if os.path.exists(h5_path):
                    h5_files[emb_type] = h5py.File(h5_path, 'r')

            # Process all mutants for this dms_id
            for sample_info in sample_group:
                mutant_id = sample_info["mutant_id"]
                label = sample_info["label"]
                
                embeddings_to_concat = []
                try:
                    for emb_type in embedding_list:
                        if emb_type == "esm_if":
                            embedding = h5_files[emb_type]["struc_repre"][()].mean(0).squeeze()
                        elif emb_type == "gearnet":
                            #pdb.set_trace()
                            embedding = h5_files[emb_type]["graph_embedding"][()].squeeze()
                        else:
                            embedding = h5_files[emb_type][mutant_id][()].squeeze()
                        embeddings_to_concat.append(embedding)
                except (KeyError, IOError):
                    continue # Skip if a mutant or file is missing

                # Concatenate embeddings and convert to tensors
                if len(embeddings_to_concat) == len(embedding_list):
                    final_embedding = np.concatenate(embeddings_to_concat, axis=-1)
                    self.loaded_data.append((
                        torch.tensor(final_embedding, dtype=torch.float32),
                        torch.tensor(label, dtype=torch.float32)
                    ))

            # Close all the opened H5 files for this group
            for hf in h5_files.values():
                hf.close()
        
        print(f"✅ Dataset initialized. Split: {split}. Number of samples loaded: {len(self.loaded_data)}")

    def __len__(self):
        # The length is the number of pre-loaded items
        return len(self.loaded_data)

    def __getitem__(self, idx):
        # Return the pre-loaded data directly
        return self.loaded_data[idx]

class ProteinGymDataset(Dataset):
    def __init__(self, data_path, embedding_path, embedding_list, split="train", split_method="random", test_fold=0, ignore_files=[], apply_layernorm=False):
        """
        PyTorch Dataset that pre-loads all protein embeddings into memory for faster training.
        """
        # --- 1. Initial Setup (your original code) ---
        ref_df = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
        self.dms_to_pdb_map = pd.Series(ref_df["pdb_file"].values, index=ref_df["DMS_id"]).to_dict()

        # --- 2. Gather Sample Metadata (your original code) ---
        all_samples_metadata = []
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f not in ignore_files]
        for filename in csv_files:
            dms_id = filename.replace(".csv", "")
            mutations_df = pd.read_csv(os.path.join(data_path, filename))
            fold_column = f"fold_{split_method}_5"
            if fold_column not in mutations_df.columns:
                raise ValueError(f"Split column '{fold_column}' not found in {filename}")

            for _, row in mutations_df.iterrows():
                all_samples_metadata.append({
                    "dms_id": dms_id,
                    "mutant_id": row["mutant"],
                    "label": row["DMS_score"],
                    "fold": row[fold_column]
                })

        # --- 2.5 ✨ Perform Per-File Standardization ---
        # Convert metadata to a DataFrame for efficient normalization
        meta_df = pd.DataFrame(all_samples_metadata)

        # Group by DMS ID and apply Z-score normalization to each group's labels
        meta_df['label'] = meta_df.groupby('dms_id')['label'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        # Handle cases where std is zero (all labels in a file are the same)
        meta_df['label'].fillna(0, inplace=True)

        # Convert back to the list of dictionaries format for the rest of your code
        all_samples_metadata = meta_df.to_dict('records')
        # --- End of new code ---

        # --- 3. Split Metadata into Train/Test (your original code) ---
        if split == "train":
            samples_to_load = [s for s in all_samples_metadata if s["fold"] != test_fold]
        else:
            samples_to_load = [s for s in all_samples_metadata if s["fold"] == test_fold]

        grouped_samples = defaultdict(list)
        for sample in samples_to_load:
            grouped_samples[sample["dms_id"]].append(sample)

        # --- 4. Pre-load all data into memory (your original code) ---
        print(f"Pre-loading {len(samples_to_load)} samples for '{split}' split into memory...")
        self.loaded_data = []

        for dms_id, sample_group in tqdm(grouped_samples.items(), desc="Loading embedding files"):
            h5_files = {}
            for emb_type in embedding_list:
                if emb_type in ["esm2", "saprot"]:
                    h5_filename = f"{dms_id}.h5"
                else: # esm_if, gearnet
                    pdb_id = self.dms_to_pdb_map.get(dms_id)
                    if not pdb_id: continue # Skip if no PDB mapping
                    pdb_id = pdb_id.split(".")[0]
                    h5_filename = f"{pdb_id}.h5"

                h5_path = os.path.join(embedding_path, emb_type, h5_filename)
                if os.path.exists(h5_path):
                    h5_files[emb_type] = h5py.File(h5_path, 'r')
            for sample_info in sample_group:
                mutant_id = sample_info["mutant_id"]
                label = sample_info["label"] # This now uses the normalized label

                embeddings_to_concat = []
                try:
                    for emb_type in embedding_list:
                        # This special logic for esm_if/gearnet is preserved
                        if emb_type == "esm_if":
                            embedding = h5_files[emb_type]["struc_repre"][()].mean(0).squeeze()
                        elif emb_type == "gearnet":
                            embedding = h5_files[emb_type]["graph_embedding"][()].squeeze()
                        else:
                            embedding = h5_files[emb_type][mutant_id][()].squeeze()
                        if apply_layernorm:
                            #print(mutant_id,emb_type, embedding.shape)
                            mu = embedding.mean()
                            std = embedding.std()
                            embedding = (embedding-mu)/std
                        embeddings_to_concat.append(embedding)
                except (KeyError, IOError):
                    continue

                if len(embeddings_to_concat) == len(embedding_list):
                    final_embedding = np.concatenate(embeddings_to_concat, axis=-1)
                    self.loaded_data.append((
                        torch.tensor(final_embedding, dtype=torch.float32),
                        torch.tensor(label, dtype=torch.float32)
                    ))

            for hf in h5_files.values():
                hf.close()

        print(f"✅ Dataset initialized. Split: {split}. Number of samples loaded: {len(self.loaded_data)}")

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        return self.loaded_data[idx]
