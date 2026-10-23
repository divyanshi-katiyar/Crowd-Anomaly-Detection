"""Training loops and utilities for Crowd Anomaly Detection models."""

import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve compute device with fallback order: preferred -> cuda -> mps -> cpu."""
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_cnn_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """Train the CNN Autoencoder on normal crowd frames.

    Args:
        model: CrowdAutoencoder instance.
        train_loader: DataLoader yielding batches of shape (B, 3, H, W).
        epochs: Number of training epochs. Defaults to 10.
        lr: Learning rate for Adam optimizer. Defaults to 1e-3.
        device: PyTorch device. If None, auto-detected.
        save_path: Optional path to save model weights .pt file.

    Returns:
        Dict containing list of loss values per epoch.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"loss": []}

    print(f"Starting CNN Autoencoder training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch in progress_bar:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

        epoch_loss = total_loss / max(1, num_batches)
        history["loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return history


def train_convlstm_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 8,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """Train the Spatio-Temporal ConvLSTM Autoencoder on sequence batches.

    Args:
        model: ConvLSTMAutoencoderV2 instance.
        train_loader: DataLoader yielding sequence batches of shape (B, T, 3, H, W).
        epochs: Number of training epochs. Defaults to 8.
        lr: Learning rate for Adam optimizer. Defaults to 1e-4.
        device: PyTorch device. If None, auto-detected.
        save_path: Optional path to save model weights .pt file.

    Returns:
        Dict containing list of loss values per epoch.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"loss": []}

    print(f"Starting ConvLSTM Autoencoder training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for seq in progress_bar:
            seq = seq.to(device)

            optimizer.zero_grad()
            recon = model(seq)
            loss = criterion(recon, seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

        epoch_loss = total_loss / max(1, num_batches)
        history["loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return history
