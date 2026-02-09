import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os,sys
import argparse
from scipy.stats import spearmanr
from model import EmbeddingMLP
import pdb
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import time
import esm
from esm import Alphabet, pretrained
def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = "auto",
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    # Check whether the structure is predicted by AlphaFold2
    if plddt_mask == "auto":
        with open(path, "r") as r:
            plddt_mask = True if "alphafold" in r.read().lower() else False
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            #pdb.set_trace()
            if plddt_mask:
                try:
                    plddts = extract_plddt(path)
                    assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
                
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Failed to mask plddt for {name}")
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict

def generate_single_embedding(mutant_sequence, pdb_path, encoders, emb_type):
    device = next(encoders["mlp"].parameters()).device
    #pdb.set_trace()
    if emb_type == "esm2":
        sequence_data = [("protein_seq", mutant_sequence)]
        tokenizer = encoders["esm2_alphabet"].get_batch_converter()
        _,_,batch_tokens = tokenizer(sequence_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = encoders["esm2_model"](batch_tokens, repr_layers=[33], return_contacts=False)
            embeddings = results["representations"][33].squeeze()[1:-1, :].mean(dim=0).cpu().numpy()
    elif emb_type == "saprot":
        foldseek_cmd_path = os.path.dirname(sys.executable)+"/foldseek" # To do
        struc_seq_dict = get_struc_seq(foldseek_cmd_path, pdb_path, ["A"], plddt_mask = False)
        struc_seq = struc_seq_dict["A"][1].lower()
        #pdb.set_trace()
        if len(mutant_sequence) != len(struc_seq):
             #raise ValueError("Length mismatch for SaProt!")
             return None
        combined_input = "".join(char for pair in zip(mutant_sequence, struc_seq) for char in pair)
        inputs = encoders["saprot_tokenizer"](combined_input, return_tensors="pt", truncation=False, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = encoders["saprot_model"](**inputs, output_hidden_states=True)
            token_embeddings = outputs.hidden_states[-1]
            token_embeddings = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed_embeddings = torch.sum(token_embeddings * mask, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            protein_embedding = summed_embeddings / summed_mask
            #pdb.set_trace()
            embeddings = protein_embedding.cpu().numpy().squeeze()
    elif emb_type == "esm_if":
        structure = esm.inverse_folding.util.load_structure(pdb_path,"A")
        coords,seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        rep = esm.inverse_folding.util.get_encoder_output(encoders["esm_if_model"],encoders["esm_if_alphabet"],coords)
        embeddings = rep.mean(0).squeeze().cpu().detach().numpy()
    elif emb_type == "gearnet":
        protein = data.Protein.from_pdb(pdb_path)
        protein_batch = data.Protein.pack([protein])
        protein_graph = encoders["gearnet_constructor"](protein_batch)
        protein_graph = protein_graph.to(device)
        with torch.no_grad():
            output = encoders["gearnet_model"](protein_graph,protein_graph.residue_feature.float())
        embeddings = output["graph_feature"].squeeze().cpu().detach().numpy()
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MLP on a DMS dataset.")
    parser.add_argument('--embedding_list', nargs='+', required=True, help="List of embeddings the MLP was trained on.")
    parser.add_argument('--test_fold', type=int, required=False, help="The test fold to evaluate.")
    parser.add_argument('--dms_csv', type=str, required=True, help="The DMS csv file of the protein to evaluate.")
    parser.add_argument('--pdb_path',type=str, required=False, help = "path to the structure file")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the trained EmbeddingMLP checkpoint.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBEDDING_DIMS = {"esm2": 1280, "esm_if": 512, "gearnet": 3072, "saprot": 1280}
    input_dim = sum(EMBEDDING_DIMS[name] for name in args.embedding_list)

    print("Loading all required models...")
    encoders = {}
    #pdb.set_trace()
    if "esm2" in args.embedding_list:
        model, alphabet = pretrained.esm2_t33_650M_UR50D()
        model.eval()
        model.to(device)
        encoders["esm2_alphabet"] = alphabet
        encoders["esm2_model"] = model
    if "saprot" in args.embedding_list:
        tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
        model = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_650M_AF2", use_safetensors=True)
        model.to(device)
        model.eval()
        encoders["saprot_tokenizer"] = tokenizer
        encoders["saprot_model"] = model
    if "esm_if" in args.embedding_list:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model.eval()
        model.to(device)
        encoders["esm_if_model"] = model
        encoders["esm_if_alphabet"] = alphabet
    if "gearnet" in args.embedding_list:
        from torchdrug import data,models,layers
        from torchdrug.layers import geometry
        model = models.GearNet(
                input_dim=21,
                hidden_dims = [512,512,512,512,512,512],
                num_relation = 7,
                edge_input_dim = 59,
                num_angle_bin = 8,
                batch_norm = True,
                concat_hidden = True,
                short_cut = True,
                readout = "sum"
                )
        CHECKPOINT_FILE = "../ckpt/mc_gearnet_edge.pth"
        state_dict = torch.load(CHECKPOINT_FILE,map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        graph_constructor = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                     edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                  geometry.KNNEdge(k=10, min_distance=5),
                                                                  geometry.SequentialEdge(max_distance=2)],
                                                     edge_feature="gearnet")
        encoders["gearnet_model"] = model
        encoders["gearnet_constructor"] = graph_constructor

    mlp_model = EmbeddingMLP(input_dim = input_dim).to(device)
    mlp_model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    mlp_model.eval()
    encoders["mlp"] = mlp_model
    print("All models loaded successfully.")

    ref_df = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
    dms_to_pdb_map = pd.Series(ref_df["pdb_file"].values, index=ref_df["DMS_id"]).to_dict()

    mutant_csv_path = args.dms_csv
    pdb_path = args.pdb_path
    if pdb_path is None:
        dms_id = mutant_csv_path.split("/")[-1].split(".")[0]
        pdb_filename = dms_to_pdb_map.get(dms_id)
        pdb_path = f"../dataset/ProteinGym/AF2_structures/{pdb_filename}"

    mutations_df = pd.read_csv(mutant_csv_path)
    test_df = mutations_df[mutations_df[f"fold_random_5"] == args.test_fold].copy()
    print(f"Evaluating {len(test_df)} mutants for {args.dms_csv} in fold {args.test_fold}...")

    # cache structure-only embeddings per PDB (and emb_type)
    struct_cache = {}
    STRUCT_ONLY = {"esm_if", "gearnet"}  # saprot depends on sequence → don't cache

    predictions = []
    multi_modal = len(args.embedding_list)>1
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {args.dms_csv}"):
        mutant_sequence = row["mutated_sequence"]
        embedding_to_concat = []
        skip_row = False

        for emb_type in args.embedding_list:
        # Use cache for structure-only embeddings
            if emb_type in STRUCT_ONLY:
                key = (emb_type, pdb_path)
                if key not in struct_cache:
                    emb = generate_single_embedding(mutant_sequence, pdb_path, encoders, emb_type=emb_type)
                # If something went wrong once, keep it as None to avoid recompute spam
                    struct_cache[key] = emb
                embedding = struct_cache[key]
            else:
            # saprot (sequence + structure) → recompute per mutant
                embedding = generate_single_embedding(mutant_sequence, pdb_path, encoders, emb_type=emb_type)

            if embedding is None:
            # length mismatch or other failure for this mutant → skip prediction
                predictions.append(None)
                skip_row = True
            # optional: log once per row
            # print(f"Skipping row due to missing {emb_type} embedding")
                break  # no need to compute other embeddings for this row
            if multi_modal:
                mu = embedding.mean()
                std = embedding.std()
                embedding = (embedding-mu)/std
            embedding_to_concat.append(embedding)

        if skip_row:
            continue

        final_embedding = torch.tensor(
            np.concatenate(embedding_to_concat),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = mlp_model(final_embedding)

        predictions.append(prediction.item())


    predictions = np.array(predictions)
    df = pd.DataFrame({'Score': predictions})
    df.to_csv(args.output_file,index = False)


    print("\n--- Evaluation Results ---")
