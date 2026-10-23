"""Configuration loading and validation utilities."""

import os
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "train_path": "data/train",
        "test_path": "data/test",
        "image_size": [256, 256],
        "sequence_length": 10,
    },
    "model_ae": {
        "in_channels": 3,
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "checkpoint_path": "results/checkpoints/autoencoder.pt",
    },
    "model_convlstm": {
        "in_channels": 3,
        "hidden_dim": 64,
        "kernel_size": 3,
        "epochs": 8,
        "batch_size": 2,
        "learning_rate": 0.0001,
        "checkpoint_path": "results/checkpoints/convlstm.pt",
    },
    "detection": {
        "gaussian_sigma": 10.0,
        "window_size": 30,
        "threshold_percentile": 95.0,
        "fusion_weight_app": 0.4,
        "fusion_weight_mot": 0.6,
    },
    "ground_truth": {
        "gt_start": 780,
        "gt_end": 830,
    },
    "output": {
        "results_dir": "results",
        "save_plots": True,
    }
}


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file, merging with default settings.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            if user_config and isinstance(user_config, dict):
                for key, val in user_config.items():
                    if isinstance(val, dict) and key in config:
                        config[key].update(val)
                    else:
                        config[key] = val
    return config
