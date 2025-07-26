"""
File: dataset.py
Author: Juan Montesinos
Created: 26/07/2025

Description:
    Photoplethysmograph (PPG) dataset for regression challenge.
    Base class for PyTorch Dataset, handling time-series data with additional features.
Usage:
    import dataset in your training script.

Copyright: (c) 2025, Juan Montesinos. All rights reserved.
"""

import json

import numpy as np
import polars as pl
import torch
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import const as C


def highpass_filter_ppg(ppg_signal, fs=C.SR, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype="high")
    return filtfilt(b, a, ppg_signal)


class BaseDataset(Dataset):
    """
    PyTorch Dataset for regression on time-series data with additional features,
    including an optional train-test split.

    Args:
        filtered (bool): If True, apply highpass_filter_ppg to each time-series.
        seed (int): Random seed for reproducibility in train-test split.
        train (bool): If True, dataset represents the training split; otherwise test split.
        test_size (float): Proportion of data to reserve for the test set.

    Attributes:
        ts_arr (Tensor): Time-series array of shape (N_split, TS_length).
        feats (Tensor): Extra features array of shape (N_split, num_feats).
        labels (Tensor): Labels array of shape (N_split,).
    """

    def __init__(
        self,
        filtered: bool = True,
        seed: int = 1995,
        train: bool = True,
        test_size: float = 0.2,
        device="cpu",
    ):
        train_data = pl.read_csv(C.TRAIN_PATH)
        labels_np = pl.read_csv(C.LABELS_PATH).to_numpy()[:, 0]

        ts_np = train_data[C.ts_columns].to_numpy()  # N x TS_length
        feats_np = train_data[C.feats_columns].to_numpy()  # N x num_feats

        if filtered:
            ts_np = np.stack([highpass_filter_ppg(x) for x in ts_np])

        ts_train, ts_test, feats_train, feats_test, y_train, y_test = train_test_split(
            ts_np,
            feats_np,
            labels_np,
            test_size=test_size,
            random_state=seed,
        )
        self.mean = ts_train.mean()
        self.std = ts_train.std()
        self.mean_labels = y_train.mean()
        self.std_labels = y_train.std()

        ts_split = ts_train if train else ts_test
        feats_split = feats_train if train else feats_test
        labels_split = y_train if train else y_test

        self.ts_arr = (torch.from_numpy((ts_split - self.mean) / self.std).float().contiguous()).to(device)
        self.feats = torch.from_numpy(feats_split).float().contiguous().to(device)
        self.labels = (
            torch.from_numpy((labels_split - self.mean_labels) / self.std_labels).float().contiguous().to(device)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x_ts = self.ts_arr[idx] 
        x_feats = self.feats[idx]
        y = self.labels[idx]
        return x_ts, x_feats, y


class TestDataset(Dataset):
    """
    PyTorch Dataset for inference on the unlabeled test set.
    Applies same preprocessing and normalization as training.
    """

    def __init__(
        self,
        csv_path: str,
        stats_path: str,
        filtered: bool = True,
        device: str = "cpu",
    ):
        # Load test data
        df = pl.read_csv(csv_path)
        ts_np = df[C.ts_columns].to_numpy()  # N x TS_length
        feats_np = df[C.feats_columns].to_numpy()  # N x num_feats

        # Optional high-pass filter
        if filtered:
            ts_np = np.stack([highpass_filter_ppg(x, fs=C.SR) for x in ts_np])

        # Load normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)
        mean = stats["data_mean"]
        std = stats["data_std"]
        self.mean_labels = stats["labels_mean"]
        self.std_labels = stats["labels_std"]

        # Normalize
        ts_norm = (ts_np - mean) / std

        # Convert to tensors
        self.ts = torch.from_numpy(ts_norm).float().to(device)
        self.feats = torch.from_numpy(feats_np).float().to(device)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx):
        return self.ts[idx], self.feats[idx]
