#!/usr/bin/env python3
"""CLI script for training Crowd Anomaly Detection models (CNN Autoencoder or ConvLSTM)."""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CrowdFrameDataset, CrowdSequenceDataset
from src.models.cnn_autoencoder import CrowdAutoencoder
from src.models.convlstm import ConvLSTMAutoencoderV2
from src.training.train import get_device, train_cnn_autoencoder, train_convlstm_autoencoder
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Crowd Anomaly Detection models."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["autoencoder", "convlstm"],
        default="autoencoder",
        help="Model architecture to train ('autoencoder' or 'convlstm')."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training frames directory (overrides config)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'mps', 'cpu', or 'auto')."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output path for model checkpoint (.pt file)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    train_path = args.data_path or config["data"]["train_path"]
    img_size = tuple(config["data"]["image_size"])
    device = get_device(args.device)

    print(f"==================================================")
    print(f"  Crowd Anomaly Detection Training")
    print(f"  Model:       {args.model}")
    print(f"  Dataset:     {train_path}")
    print(f"  Image Size:  {img_size}")
    print(f"  Device:      {device}")
    print(f"==================================================")

    if not os.path.exists(train_path):
        print(f"Error: Training directory '{train_path}' does not exist.")
        print("Please check dataset path or provide --data_path /path/to/train_frames")
        sys.exit(1)

    if args.model == "autoencoder":
        cfg = config.get("model_ae", {})
        epochs = args.epochs or cfg.get("epochs", 10)
        batch_size = args.batch_size or cfg.get("batch_size", 8)
        lr = args.lr or cfg.get("learning_rate", 0.001)
        save_path = args.save_path or cfg.get("checkpoint_path", "results/checkpoints/crowd_autoencoder.pt")

        dataset = CrowdFrameDataset(train_path, resize=img_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = CrowdAutoencoder(in_channels=cfg.get("in_channels", 3))

        train_cnn_autoencoder(
            model=model,
            train_loader=loader,
            epochs=epochs,
            lr=lr,
            device=device,
            save_path=save_path
        )

    elif args.model == "convlstm":
        cfg = config.get("model_convlstm", {})
        epochs = args.epochs or cfg.get("epochs", 8)
        batch_size = args.batch_size or cfg.get("batch_size", 2)
        lr = args.lr or cfg.get("learning_rate", 0.0001)
        seq_len = config["data"].get("sequence_length", 10)
        save_path = args.save_path or cfg.get("checkpoint_path", "results/checkpoints/convlstm_autoencoder.pt")

        dataset = CrowdSequenceDataset(train_path, seq_len=seq_len, resize=img_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = ConvLSTMAutoencoderV2(
            in_channels=cfg.get("in_channels", 3),
            hidden_dim=cfg.get("hidden_dim", 64),
            kernel_size=cfg.get("kernel_size", 3)
        )

        train_convlstm_autoencoder(
            model=model,
            train_loader=loader,
            epochs=epochs,
            lr=lr,
            device=device,
            save_path=save_path
        )


if __name__ == "__main__":
    main()
