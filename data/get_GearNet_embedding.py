import torch
from torchdrug import data, models, layers
from torchdrug.layers import geometry
import os
import h5py
from tqdm import tqdm

if __name__=='__main__':
    PDB_FOLDER = "../dataset/ProteinGym/AF2_structures/"
    RESULT_PATH = "../dataset/ProteinGym/representation/gearnet/"
    CHECKPOINT_FILE = "../ckpt/mc_gearnet_edge.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULT_PATH, exist_ok=True)

    gearnet_model = models.GearNet(
        input_dim=21,
        hidden_dims=[512, 512, 512, 512, 512, 512],
        num_relation=7,
        edge_input_dim=59,
        num_angle_bin=8,
        batch_norm=True,
        concat_hidden=True,
        short_cut=True,
        readout="sum"
    )
    state_dict = torch.load(CHECKPOINT_FILE,map_location="cpu")
    gearnet_model.load_state_dict(state_dict)
    gearnet_model.to(DEVICE)
    gearnet_model.eval()
    graph_constructor = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                     edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                  geometry.KNNEdge(k=10, min_distance=5),
                                                                  geometry.SequentialEdge(max_distance=2)],
                                                     edge_feature="gearnet")

    pdb_files = [f for f in os.listdir(PDB_FOLDER) if f.endswith('.pdb')]
    print(f"\nFound {len(pdb_files)} PDB files. Starting embedding generation...")

    for pdb_filename in tqdm(pdb_files):
        pdb_path = os.path.join(PDB_FOLDER,pdb_filename)
        protein = data.Protein.from_pdb(pdb_path)
    
    # Pack the protein into a batch and apply the graph construction
        protein_batch = data.Protein.pack([protein])
        protein_graph = graph_constructor(protein_batch)
        protein_graph = protein_graph.to(DEVICE)
        with torch.no_grad():
            output = gearnet_model(protein_graph,protein_graph.residue_feature.float())
            node_embedding = output["node_feature"]
            graph_embedding = output["graph_feature"]
        output_filename = os.path.splitext(pdb_filename)[0] + ".h5"
        output_path = os.path.join(RESULT_PATH,output_filename)
        
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset('node_embedding', data=node_embedding.cpu().numpy())
            hf.create_dataset('graph_embedding', data=graph_embedding.cpu().numpy())
