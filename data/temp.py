import torch
from esm import Alphabet, pretrained
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

def esm_encode(model, repr_layers, tokenizer, seq):
    """Encodes a single sequence using the ESM model."""
    device = next(model.parameters()).device
    data = [("protein_seq", seq)]
    _, _, batch_tokens = tokenizer(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        result = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)
    # Return logits and the sequence representation (excluding start/end tokens)
    return result["logits"].squeeze(), result["representations"][repr_layers[0]].squeeze()[1:-1, :]

if __name__ == "__main__":
    # --- Configuration ---
    reference_file_path = "../dataset/ProteinGym/reference_files/DMS_substitutions.csv"
    data_path = "../dataset/ProteinGym/substitution/"
    benchmark_output_file = "esm2_fast_benchmark.csv"
    
    # 👈 How many mutants to test from each file.
    # A smaller number is faster; 30-50 is usually a good balance.
    NUM_SAMPLES_PER_FILE = 20 
    # -------------------

    # --- Set up ESM model ---
    print("Loading ESM2 model...")
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval()
    repr_layers = [33]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    print(f"Model loaded on device: {device}")

    # --- Main Processing Loop ---
    reference_file = pd.read_csv(reference_file_path)
    timing_results = []

    for index, row in reference_file.iterrows():
        dms_id = row["DMS_id"]
        seq_len = row["seq_len"]
        
        mutant_csv_path = os.path.join(data_path, dms_id + ".csv")
        if not os.path.exists(mutant_csv_path):
            print(f"Warning: File not found for {dms_id}, skipping.")
            continue
            
        mt_seq_list = pd.read_csv(mutant_csv_path)
        total_num_mutants = len(mt_seq_list)
        
        if total_num_mutants == 0:
            print(f"Warning: File for {dms_id} is empty, skipping.")
            continue

        # --- Create a sample of mutants to test ---
        # If the file has fewer mutants than our sample size, just use all of them.
        actual_sample_size = min(total_num_mutants, NUM_SAMPLES_PER_FILE)
        sample_df = mt_seq_list.sample(n=actual_sample_size, random_state=42) # Use a seed for reproducibility

        print(f"\nBenchmarking {dms_id}: Testing {actual_sample_size} of {total_num_mutants} total mutants...")
        
        # --- Timing block for the SAMPLE ---
        start_time = time.time()

        # We only need to run the encoding, no need to save the results for the benchmark
        for _, mt_row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"{dms_id} (sample)"):
            mt_seq = mt_row["mutated_sequence"]
            esm_encode(model, repr_layers, batch_converter, mt_seq)
        
        end_time = time.time()
        # --- End of timing block ---

        # --- Calculate, Extrapolate, and Store Results ---
        sample_processing_time = end_time - start_time
        
        # Calculate the average time per mutant based ONLY on the sample
        avg_time_per_mutant = sample_processing_time / actual_sample_size
        
        # ✨ Extrapolate to estimate the time for the FULL file
        estimated_full_time = avg_time_per_mutant * total_num_mutants

        print(f"Sample processed in {sample_processing_time:.2f}s. Estimated full file time: {estimated_full_time:.2f}s")
        
        timing_results.append({
            "DMS_id": dms_id,
            "seq_len": seq_len,
            "num_mutants": total_num_mutants,
            "total_aas": total_num_mutants * seq_len,
            "estimated_total_time_s": estimated_full_time, # 👈 USE THIS FOR PARTITIONING
            "sample_size_tested": actual_sample_size,
            "avg_time_s_per_mutant": avg_time_per_mutant
        })

    # --- Save Benchmark Report ---
    print(f"\n✅ Benchmarking complete. Saving report to {benchmark_output_file}...")
    benchmark_df = pd.DataFrame(timing_results)
    benchmark_df.to_csv(benchmark_output_file, index=False)
    print("Report saved successfully.")
    print(benchmark_df.head())
