import torch
from esm import Alphabet, pretrained
from scipy.stats import spearmanr

import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import os
def convert_mutation_string(hgvs_string):
    """
    Converts a protein mutation string from HGVS p.Trp113Ala format to W113A format.

    Args:
        hgvs_string: The input mutation string in HGVS format (e.g., 'p.Trp113Ala').

    Returns:
        The converted mutation string in one-letter code format (e.g., 'W113A'),
        or None if the input format is invalid.
    """
    # Dictionary to map three-letter amino acid codes to one-letter codes
    amino_acid_map = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

    # Remove the 'p.' prefix
    if hgvs_string.startswith('p.'):
        hgvs_string = hgvs_string[2:]

    # Use regular expression to parse the components: original_aa, position, new_aa
    import re
    match = re.match(r'([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', hgvs_string)

    if match:
        original_aa_3letter = match.group(1)
        position = match.group(2)
        new_aa_3letter = match.group(3)

        # Convert the three-letter codes to one-letter codes
        try:
            original_aa_1letter = amino_acid_map[original_aa_3letter]
            new_aa_1letter = amino_acid_map[new_aa_3letter]
        except KeyError:
            return None # Handle cases where amino acid code is not in the map

        # Combine the parts into the new format
        return f"{original_aa_1letter}{position}{new_aa_1letter}"
    else:
        return None # Return None for strings that don't match the expected format



def esm_encode(model,repr_layers,tokenizer,seq):
    device = next(model.parameters()).device
    data = [("wt_protein",seq)]
    _,_,batch_tokens = tokenizer(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        result = model(batch_tokens, repr_layers, return_contacts=False)
    return result["logits"].squeeze(),result["representations"][repr_layers[0]].squeeze()[1:-1,:]

def wt_marginals(alphabet,mt_seq_list,logits):
    score_dict = {}
    for index,row in mt_seq_list.iterrows():
        mutant = row["mutant"]
        #mutant = convert_mutation_string(mutant)
        if ":" not in mutant:
            wt,idx,mt = mutant[2:4],int(mutant[1:-1]),mutant[-1]
        else:
            continue #only consider single site mutation for now
        wt_token,mt_token = alphabet.get_idx(wt),alphabet.get_idx(mt)
        pred_score = logits[idx-1,mt_token]-logits[idx-1,wt_token]
        score_dict[mutant] = [pred_score.detach().cpu().numpy()]
    return score_dict

def masked_marginals(model,repr_layers,alphabet,wt_seq,mt_seq_list):
    score_dict = {}

    device = next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1",wt_seq)]
    _,_,batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    all_token_probs = []
    for i in tqdm(range(batch_tokens.size(1))):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0,i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = model(batch_tokens_masked)["logits"]
        all_token_probs.append(token_probs[:,i])
        masked_token_probs = torch.cat(all_token_probs,dim=0)
    for index,row in mt_seq_list.iterrows():
        mutant = row["mutant"]
        if ":" not in mutant:
            wt,idx,mt = mutant[0],int(mutant[1:-1]),mutant[-1]
        else:
            continue
        wt_token,mt_token = alphabet.get_idx(wt),alphabet.get_idx(mt)
        pred_score = masked_token_probs[idx,mt_token]-masked_token_probs[idx,wt_token]
        score_dict[mutant]=[pred_score.detach().cpu().numpy()]
    return score_dict
    
if __name__ == "__main__":
    reference_file = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
    data_path = "../dataset/ProteinGym/substitution/"
    
    #set up esm model
    model,alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval() #disable dropout for deterministic results
    repr_layers = [33]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    
    wt_spearman_list = []
    masked_spearman_list = []
    #for index, row in reference_file.iterrows():
    dms_id = "A0A1I9GEU1_NEIME_Kennouche_2019"
    wt_seq = "FTLIELMIVIAIVGILAAVALPAYQDYTARAQVSEAILLAEGQKSAVTEYYLNHGEWPGDNSSAGVATSADIKGKYVQSVTVANGVITAQMASSNVNNEIKSKKLSLWAKRQNGSVKWFCGQPVTRTTATATDVAAANGKTDDKINTKHLPSTCRDDSSAS"
    seq_len = len(wt_seq)
    mt_seq_list = pd.read_csv(os.path.join(data_path,dms_id+".csv"))
    logits,representation = esm_encode(model,repr_layers,batch_converter,wt_seq)
    score_dict = wt_marginals(alphabet,mt_seq_list,logits)
    #pdb.set_trace()
    pred_score = [value_list[0] for value_list in score_dict.values()]
    #gt_score = [value_list[1] for value_list in score_dict.values()]
    gt_score = mt_seq_list["DMS_score"]
    wt_marginals_spearman = spearmanr(pred_score,gt_score)[0]
    #wt_spearman_list.append(wt_marginals_spearman)
    
        #score_dict = masked_marginals(model,repr_layers,alphabet,wt_seq,mt_seq_list)
        #pred_score = [value_list[0] for value_list in score_dict.values()]
        #gt_score = [value_list[1] for value_list in score_dict.values()]
        #masked_marginals_spearman = spearmanr(pred_score,gt_score)[0]
        #masked_spearman_list.append(masked_marginals_spearman)
    print(dms_id,wt_marginals_spearman)
