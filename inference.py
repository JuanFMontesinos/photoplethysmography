"""
File: inference.py
Author: Juan Montesinos
Created: 26/07/2025

Description:
    Inference script for running the trained PPG model on the test set and saving predictions.
Usage:
    uv run inference.py
"""

import os
import numpy as np
import shutil
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import const as C
from dataset import TestDataset
from train_naive_solution import Model

if __name__ == "__main__":
    pred_dir = 'deliverable'
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    model_dir = "training"
    model_path = os.path.join(model_dir, "best_model.pth")
    stats_path = os.path.join(model_dir, "preprocessing_stats.json")
    test_csv = C.TEST_PATH
    output_file = f"{pred_dir}/predictions.txt"

    # Load model
    model = torch.compile(Model().to(device))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare test dataset and loader
    test_ds = TestDataset(test_csv, stats_path, filtered=True, device=device)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Run inference
    all_preds = []
    with torch.no_grad():
        for ts_batch, feats_batch in loader:
            preds = model(ts_batch, feats_batch)
            # Un-normalize
            preds = preds * test_ds.std_labels + test_ds.mean_labels
            all_preds.extend(preds.cpu().numpy().tolist())

    # Save predictions
    
    with open(output_file, "w") as f:
        for p in all_preds:
            f.write(f"{p}\n")
    labels_arr = np.array(all_preds)
    plt.figure()
    plt.hist(labels_arr.flatten(), bins=100)
    plt.title("Histogram of predictions")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f"{pred_dir}/labels_hist.png")
    print(f"Saved {len(all_preds)} predictions to '{output_file}'")
