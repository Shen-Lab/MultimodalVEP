import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import h5py
import os
import argparse
import time
from dataset import ProteinGymDataset
from model import EmbeddingMLP
import pdb

if __name__ == "__main__":
    # --- 3. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Train an MLP on pre-computed protein embeddings.")
    parser.add_argument('--embedding_list', nargs='+', required=True, help="List of embeddings to use (e.g., esm2 gearnet)")
    parser.add_argument('--test_fold', type=int, required=True, help="Fold to use for testing (0-4)")
    parser.add_argument('--ckpt_path', type=str, default="checkpoints/", help="Directory to save model checkpoints.")
    parser.add_argument('--apply_layernorm', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # --- 4. Configuration and Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_path, exist_ok=True)

    # Define the dimensions of your embeddings.
    # IMPORTANT: Verify these match your saved embedding files!
    EMBEDDING_DIMS = {
        "esm2": 1280,
        "esm_if": 512,
        "gearnet": 3072, # This is for a 6-layer concatenated graph embedding
        "saprot": 1280
    }

    # Calculate the input dimension for the MLP based on the user's choice
    input_dim = sum(EMBEDDING_DIMS[name] for name in args.embedding_list)
    print(f"Using embeddings: {args.embedding_list}")
    print(f"Total embedding dimension: {input_dim}")

    # --- 5. Create Datasets and DataLoaders ---
    data_path = "../dataset/ProteinGym/substitution_split/"
    embedding_path = "../dataset/ProteinGym/representation"

    train_dataset = ProteinGymDataset(
        data_path=data_path,
        embedding_path=embedding_path,
        embedding_list=args.embedding_list,
        split="train",
        test_fold=args.test_fold,
        apply_layernorm=args.apply_layernorm
    )
    test_dataset = ProteinGymDataset(
        data_path=data_path,
        embedding_path=embedding_path,
        embedding_list=args.embedding_list,
        split="test",
        test_fold=args.test_fold,
        apply_layernorm=args.apply_layernorm
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # --- 6. Initialize Model, Optimizer, and Loss Function ---
    model = EmbeddingMLP(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # --- 7. Training and Evaluation Loop ---
    best_val_loss = float("inf")

    for epoch in range(1000):
        start_time = time.time()
        model.train()
        train_loss = 0

        for embeddings, labels in train_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            #print(loss)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for embeddings, labels in test_dataloader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                val_loss += loss_fn(outputs, labels).item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(test_dataloader)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1:03d} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f"mlp_epoch_{epoch+1}_fold{args.test_fold}.pt"))

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f"mlp_best_fold{args.test_fold}.pt"))
            print(f"  -> New best model saved with validation loss: {best_val_loss:.6f}")
