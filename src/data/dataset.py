"""PyTorch Dataset implementations for single frame and spatio-temporal sequence loading."""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrowdFrameDataset(Dataset):
    """Dataset for loading individual crowd surveillance video frames.

    Args:
        folder: Path to folder containing video frame images (.png, .jpg, .jpeg).
        resize: Target resolution as (width, height). Defaults to (256, 256).
    """

    def __init__(
        self,
        folder: str,
        resize: Tuple[int, int] = (256, 256)
    ) -> None:
        super().__init__()
        self.folder = folder
        self.resize = resize

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

        valid_exts = (".png", ".jpg", ".jpeg")
        self.files: List[str] = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(valid_exts)
        ])

        if len(self.files) == 0:
            raise ValueError(f"No image files found in folder: {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = os.path.join(self.folder, self.files[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image at: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32) / 255.0
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        return tensor


class CrowdSequenceDataset(Dataset):
    """Dataset for loading sliding-window temporal frame sequences for ConvLSTM models.

    Args:
        folder: Path to folder containing video frame images (.png, .jpg, .jpeg).
        seq_len: Temporal window length (number of consecutive frames). Defaults to 10.
        resize: Target resolution as (width, height). Defaults to (256, 256).
    """

    def __init__(
        self,
        folder: str,
        seq_len: int = 10,
        resize: Tuple[int, int] = (256, 256)
    ) -> None:
        super().__init__()
        self.folder = folder
        self.seq_len = seq_len
        self.resize = resize

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

        valid_exts = (".png", ".jpg", ".jpeg")
        self.files: List[str] = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(valid_exts)
        ])

        if len(self.files) <= self.seq_len:
            raise ValueError(
                f"Number of frames ({len(self.files)}) in {folder} "
                f"must be strictly greater than sequence length ({self.seq_len})"
            )

    def __len__(self) -> int:
        return len(self.files) - self.seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq: List[torch.Tensor] = []
        for i in range(idx, idx + self.seq_len):
            img_path = os.path.join(self.folder, self.files[i])
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image at: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.resize)
            img = img.astype(np.float32) / 255.0
            tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            seq.append(tensor)

        return torch.stack(seq)  # Shape: [T, C, H, W]
