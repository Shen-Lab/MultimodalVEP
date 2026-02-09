import torch
import esm.inverse_folding
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import h5py

if __name__ == "__main__":
    # --- Configuration ---
    data_path = "../dataset/ProteinGym/AF2_structures/"
    result_path = "../dataset/ProteinGym/representation/esm_if/"
    os.makedirs(result_path, exist_ok=True)

    # --- Set up ESM model ---
    print("Loading ESMIF model...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()  # Disable dropout
    repr_layers = [33]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on device: {device}")

    # --- Main Processing Loop ---
    prot_struc_list = [os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith('.pdb')]

    print(f"This job will process {len(prot_struc_list)} files.")

    # Iterate through the DMS IDs assigned to this job
    for prot_struc in prot_struc_list:
        structure = esm.inverse_folding.util.load_structure(prot_struc, "A")
        coords,seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        rep = esm.inverse_folding.util.get_encoder_output(model,alphabet,coords)
        output_filename = prot_struc.split("/")[-1].split(".")[0]+".h5"
        print(output_filename)
        output_path = os.path.join(result_path,output_filename)
        with h5py.File(output_path,"w") as hf:
            embedding_numpy = rep.detach().cpu().numpy()
            hf.create_dataset('struc_repre',data=embedding_numpy)
    print("Job finished. All assigned files have been processed.")
