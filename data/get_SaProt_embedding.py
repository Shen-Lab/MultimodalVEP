import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import h5py

def parse_foldseek_fasta(fasta_path):
    """
    Reads a Foldseek 3Di FASTA file and returns a dictionary mapping
    protein IDs to their structural sequences.
    """
    structure_sequences = {}
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Extracts the base ID, e.g., 'A0A140D2T1_ZIKV' from '>A0A140D2T1_ZIKV.pdb_A'
                protein_id = line.split('.')[0][1:] 
                current_id = protein_id
                structure_sequences[current_id] = ""
            else:
                structure_sequences[current_id] += line
    print(f"Loaded {len(structure_sequences)} structural sequences from FASTA file.")
    return structure_sequences

def get_saprot_embedding(sequence, structure, tokenizer, model, device):
    """
    Generates a protein-level embedding from a combined sequence and structure sequence.
    """
    # 1. Combine sequence and structure with the tokenizer's separator token
    # The model expects a format like: [SEQUENCE] [SEP] [STRUCTURE_SEQUENCE]
    combined_input = "".join(char for pair in zip(sequence, structure) for char in pair)
    print(combined_input)

    # 2. Tokenize the combined input
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 3. Get model output
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 4. Extract and mean-pool the token embeddings
    token_embeddings = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed_embeddings = torch.sum(token_embeddings * mask, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    protein_embedding = summed_embeddings / summed_mask

    return protein_embedding.cpu().numpy()

if __name__ == "__main__":
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Generate SaProt embeddings using a file list.")
    parser.add_argument(
        '--file_list',
        type=str,
        required=True,
        help="Path to a text file containing the DMS_id.csv names to process, one per line."
    )
    args = parser.parse_args()

    # --- Configuration ---
    FASTA_FILE = "../dataset/ProteinGym/foldseek_structure_token.fasta" 
    MUTANT_CSV_DIR = "../dataset/ProteinGym/substitution/"          
    RESULT_PATH = "../dataset/ProteinGym/representation/SaProt/"
    REFERENCE_FILE = "../dataset/ProteinGym/reference_files/DMS_substitutions.csv"
    # -------------------

    os.makedirs(RESULT_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load SaProt Model and Tokenizer
    print("Loading SaProt model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
    model = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_650M_AF2", use_safetensors=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # 3. Load the structural sequences from the Foldseek FASTA
    structural_data = parse_foldseek_fasta(FASTA_FILE)
    
    # --- 4. Read the list of files to process for this specific job ---
    with open(args.file_list, 'r') as f:
        # Read lines and strip any whitespace/newlines
        dms_filenames = [line.strip() for line in f.readlines() if line.strip()]
        # Remove the '.csv' extension to get the DMS_id, e.g., 'A0A140D2T1_ZIKV'
        dms_ids_to_process = [filename.replace('.csv', '') for filename in dms_filenames]
    
    print(f"\nThis job will process {len(dms_ids_to_process)} files from '{args.file_list}'.")
    
    pd_reference = pd.read_csv(REFERENCE_FILE)
    dms_to_pdb_map = pd.Series(pd_reference["pdb_file"].values,index=pd_reference["DMS_id"]).to_dict()
    # 5. Main processing loop
    for dms_id in tqdm(dms_ids_to_process, desc="Processing DMS Datasets"):
        mutant_csv_path = os.path.join(MUTANT_CSV_DIR, dms_id + ".csv")
        pdb_id = dms_to_pdb_map[dms_id].split(".")[0]
        structure_sequence = structural_data.get(pdb_id)
        sequence_length = pd_reference[pd_reference["DMS_id"]==dms_id]["seq_len"].iloc[0]
        if len(structure_sequence) != sequence_length:
            print(f"{dms_id} length mismatch")
            continue
        # Check for required files
        if not os.path.exists(mutant_csv_path):
            print(f"\nWarning: Mutant CSV for {dms_id} not found. Skipping.")
            continue
        if not structure_sequence:
            print(f"\nWarning: Structural sequence for {dms_id} not found in FASTA. Skipping.")
            continue

        # Prepare to save embeddings
        output_h5_path = os.path.join(RESULT_PATH, dms_id + ".h5")
        
        with h5py.File(output_h5_path, 'w') as hf:
            mutant_df = pd.read_csv(mutant_csv_path)
            
            for index, row in mutant_df.iterrows():
                mutant_id = row["mutant"]
                aa_sequence = row["mutated_sequence"]
                #print(len(structure_sequence),len(aa_sequence))

                embedding = get_saprot_embedding(aa_sequence, structure_sequence, tokenizer, model, device)
                hf.create_dataset(mutant_id, data=embedding)

    print("\n✅ Pipeline finished. All assigned files have been processed.")
