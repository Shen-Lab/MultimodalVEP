import torch
from esm import Alphabet, pretrained
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import h5py

def esm_encode(model, repr_layers, tokenizer, seq):
    """Encodes a single sequence using the ESM model."""
    device = next(model.parameters()).device
    # The sequence name "protein_seq" is arbitrary
    data = [("protein_seq", seq)]
    _, _, batch_tokens = tokenizer(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)
        
    # Squeeze to remove batch dimension and remove start/end tokens from representation
    logits = results["logits"].squeeze()
    representation = results["representations"][repr_layers[0]].squeeze()[1:-1, :]
    return logits, representation

def process_dms_file(dms_id, model, repr_layers, batch_converter, data_path, result_path):
    """
    Processes all mutants for a single DMS_id, saving embeddings to an HDF5 file.
    """
    mutant_csv_path = os.path.join(data_path, dms_id + ".csv")
    if not os.path.exists(mutant_csv_path):
        print(f"Warning: File not found for {dms_id}, skipping.")
        return

    mt_seq_list = pd.read_csv(mutant_csv_path)
    num_mutants = len(mt_seq_list)
    
    output_h5_path = os.path.join(result_path, dms_id + ".h5")
    
    print(f"\nProcessing {dms_id}: Saving {num_mutants} mutants to {output_h5_path}")
    with h5py.File(output_h5_path, 'w') as hf:
        # Loop through all mutant sequences in the file
        for i, (_, mt_row) in enumerate(tqdm(mt_seq_list.iterrows(), total=num_mutants, desc=dms_id)):
            mutant_id = mt_row["mutant"]
            mt_seq = mt_row["mutated_sequence"]
            _, mt_repr = esm_encode(model, repr_layers, batch_converter, mt_seq)
            embedding_numpy = mt_repr.mean(dim=0).detach().cpu().numpy()
            hf.create_dataset(mutant_id, data=embedding_numpy)
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESM2 embeddings for a list of protein files.")
    parser.add_argument(
        '--file_list',
        type=str,
        required=True,
        help="Path to a text file containing the DMS_id names to process, one per line."
    )
    args = parser.parse_args()

    # --- Configuration ---
    data_path = "../dataset/ProteinGym/substitution/"
    result_path = "../dataset/ProteinGym/representation/esm2/"
    os.makedirs(result_path, exist_ok=True)

    # --- Set up ESM model ---
    print("Loading ESM2 model...")
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval()  # Disable dropout
    repr_layers = [33]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    print(f"Model loaded successfully on device: {device}")

    # --- Main Processing Loop ---
    with open(args.file_list, 'r') as f:
        # Read lines and strip any whitespace/newlines
        dms_ids_to_process = [line.strip() for line in f.readlines()]
        # Remove the '.csv' extension to get the DMS_id
        dms_ids_to_process = [dms_id.replace('.csv', '') for dms_id in dms_ids_to_process if dms_id]

    print(f"This job will process {len(dms_ids_to_process)} files from '{args.file_list}'.")

    # Iterate through the DMS IDs assigned to this job
    for dms_id in dms_ids_to_process:
        process_dms_file(dms_id, model, repr_layers, batch_converter, data_path, result_path)

    print("Job finished. All assigned files have been processed.")
