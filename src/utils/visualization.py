"""Visualization utilities for anomaly detection curves, ground truth overlays, and frame-level heatmaps."""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.preprocessing import compute_farneback_optical_flow


def plot_anomaly_scores(
    scores: Union[List[float], np.ndarray],
    threshold: Optional[float] = None,
    threshold_label: str = "Anomaly Threshold",
    title: str = "Crowd Anomaly Score Over Time",
    xlabel: str = "Frame Index",
    ylabel: str = "Anomaly Score",
    save_path: Optional[str] = None
) -> None:
    """Plot anomaly score time-series with optional decision threshold.

    Args:
        scores: 1D sequence of scores.
        threshold: Optional threshold float value to draw as horizontal dashed line.
        threshold_label: Legend label for threshold.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save image file.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="Anomaly Score", color="#1f77b4", linewidth=1.5)

    if threshold is not None:
        plt.axhline(threshold, color="#d62728", linestyle="--", label=threshold_label)

    plt.title(title, fontsize=13)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_detection_vs_ground_truth(
    scores: np.ndarray,
    threshold: float,
    pred_segments: List[int],
    gt_start: int,
    gt_end: int,
    window: int = 30,
    title: str = "Model Detection vs Ground Truth",
    save_path: Optional[str] = None
) -> None:
    """Plot anomaly scores overlaid with predicted anomaly regions and ground-truth intervals.

    Args:
        scores: 1D anomaly score sequence.
        threshold: Anomaly cutoff threshold.
        pred_segments: List of start frame indices for detected windows.
        gt_start: Ground-truth anomaly start frame.
        gt_end: Ground-truth anomaly end frame.
        window: Sliding window length in frames.
        title: Plot title.
        save_path: Optional file path to save figure.
    """
    plt.figure(figsize=(12, 4))

    # Anomaly score curve
    plt.plot(scores, label="Final Anomaly Score", color="#1f77b4", linewidth=1.5)

    # Threshold
    plt.axhline(threshold, color="#d62728", linestyle="--", label=f"Threshold ({threshold:.3f})")

    # Predicted anomaly spans
    first_pred = True
    for idx in pred_segments:
        end_idx = min(idx + window, len(scores))
        plt.axvspan(
            idx,
            end_idx,
            color="#ff7f0e",
            alpha=0.2,
            label="Predicted Anomaly Window" if first_pred else None
        )
        first_pred = False

    # Ground truth span
    plt.axvspan(
        gt_start,
        gt_end,
        color="#2ca02c",
        alpha=0.3,
        label=f"Ground Truth ({gt_start}-{gt_end})"
    )

    plt.title(title, fontsize=13)
    plt.xlabel("Frame Index", fontsize=11)
    plt.ylabel("Anomaly Score", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_model_comparison(
    aligned_scores_dict: Dict[str, np.ndarray],
    aligned_gt: Optional[int] = None,
    title: str = "Model Comparison: Crowd Anomaly Detection",
    save_path: Optional[str] = None
) -> None:
    """Plot multiple model anomaly scores on a single aligned timeline.

    Args:
        aligned_scores_dict: Dict mapping model names (e.g. 'AE (Appearance)', 'ConvLSTM') to 1D arrays.
        aligned_gt: Optional ground-truth onset frame index.
        title: Plot title.
        save_path: Optional file path to save figure.
    """
    plt.figure(figsize=(14, 5))

    styles = {
        "AE (Appearance)": {"color": "#1f77b4", "alpha": 0.6, "linewidth": 1.2},
        "Flow (Motion)": {"color": "#ff7f0e", "alpha": 0.6, "linewidth": 1.2},
        "AE + Flow (Fusion)": {"color": "#2ca02c", "alpha": 0.9, "linewidth": 2.0},
        "ConvLSTM": {"color": "#d62728", "alpha": 0.9, "linewidth": 2.0},
    }

    for name, score in aligned_scores_dict.items():
        style = styles.get(name, {"alpha": 0.8, "linewidth": 1.5})
        plt.plot(score, label=name, **style)

    if aligned_gt is not None:
        plt.axvline(aligned_gt, color="#000000", linestyle="--", linewidth=1.5, label="GT Onset")

    plt.legend(loc="upper right")
    plt.title(title, fontsize=14)
    plt.xlabel("Frame Index", fontsize=11)
    plt.ylabel("Normalized Anomaly Score", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_qualitative_explanations(
    dataset,
    frames_to_explain: Dict[str, int],
    save_path: Optional[str] = None
) -> None:
    """Generate a multi-row visual explanation showing raw frames and Farneback motion heatmaps.

    Args:
        dataset: PyTorch Dataset of normalized frame tensors.
        frames_to_explain: Dict mapping stage label (e.g. 'Normal', 'Onset') to frame index.
        save_path: Optional file path to save figure.
    """
    num_stages = len(frames_to_explain)
    fig, axes = plt.subplots(num_stages, 2, figsize=(14, 3 * num_stages))

    if num_stages == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (label, frame_idx) in enumerate(frames_to_explain.items()):
        # Retrieve raw frame
        curr_tensor = dataset[frame_idx]
        if isinstance(curr_tensor, torch.Tensor):
            curr_np = curr_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            curr_np = curr_tensor

        # Retrieve previous frame for motion estimation
        prev_idx = max(0, frame_idx - 1)
        prev_tensor = dataset[prev_idx]
        if isinstance(prev_tensor, torch.Tensor):
            prev_np = prev_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            prev_np = prev_tensor

        # Compute optical flow magnitude
        g1 = cv2.cvtColor((prev_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor((curr_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mag, _ = compute_farneback_optical_flow(g1, g2)

        # Plot raw frame
        axes[row_idx, 0].imshow(curr_np)
        axes[row_idx, 0].set_title(f"{label} Stage - Raw Frame ({frame_idx})", fontsize=11)
        axes[row_idx, 0].axis("off")

        # Plot motion heatmap
        im = axes[row_idx, 1].imshow(mag, cmap="hot")
        axes[row_idx, 1].set_title(f"{label} Stage - Motion Magnitude", fontsize=11)
        axes[row_idx, 1].axis("off")

    plt.suptitle("Frame-level Qualitative Motion Explanation", fontsize=15, y=0.99)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
