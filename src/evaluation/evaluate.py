"""Evaluation metrics, anomaly detection scoring, and thresholding algorithms."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.preprocessing import smooth_scores, standardize_scores


def compute_reconstruction_errors(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Compute frame-wise Mean Squared Error (MSE) reconstruction errors for a CNN Autoencoder.

    Args:
        model: Trained CrowdAutoencoder.
        dataloader: DataLoader yielding batches of shape (B, 3, H, W).
        device: Device to run evaluation on.

    Returns:
        np.ndarray: Array of frame-level reconstruction error scores.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    errors: List[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing AE reconstruction error"):
            batch = batch.to(device)
            recon = model(batch)
            # Frame-by-frame MSE
            if batch.size(0) == 1:
                err = torch.mean((batch - recon) ** 2).item()
                errors.append(err)
            else:
                batch_errs = torch.mean((batch - recon) ** 2, dim=[1, 2, 3]).cpu().tolist()
                errors.extend(batch_errs)

    return np.array(errors, dtype=np.float64)


def compute_sequence_reconstruction_errors(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Compute sequence-wise Mean Squared Error (MSE) reconstruction errors for ConvLSTM.

    Args:
        model: Trained ConvLSTMAutoencoderV2.
        dataloader: DataLoader yielding sequence batches of shape (B, T, 3, H, W).
        device: Device to run evaluation on.

    Returns:
        np.ndarray: Array of sequence reconstruction error scores.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    errors: List[float] = []

    with torch.no_grad():
        for seq in tqdm(dataloader, desc="Computing ConvLSTM sequence error"):
            seq = seq.to(device)
            recon = model(seq)
            if seq.size(0) == 1:
                err = torch.mean((seq - recon) ** 2).item()
                errors.append(err)
            else:
                batch_errs = torch.mean((seq - recon) ** 2, dim=[1, 2, 3, 4]).cpu().tolist()
                errors.extend(batch_errs)

    return np.array(errors, dtype=np.float64)


def compute_hybrid_fusion(
    smooth_ae_scores: np.ndarray,
    smooth_flow_scores: np.ndarray,
    weight_app: float = 0.4,
    weight_mot: float = 0.6,
    sigma: float = 10.0
) -> np.ndarray:
    """Fuse appearance (AE) and motion (Optical Flow) scores with adaptive weighting.

    Args:
        smooth_ae_scores: Smoothed appearance anomaly scores.
        smooth_flow_scores: Smoothed optical flow motion anomaly scores.
        weight_app: Weight for appearance score. Defaults to 0.4.
        weight_mot: Weight for motion score. Defaults to 0.6.
        sigma: Smoothing parameter for post-fusion signal. Defaults to 10.0.

    Returns:
        np.ndarray: Fused hybrid anomaly scores.
    """
    app = standardize_scores(smooth_ae_scores)
    mot = standardize_scores(smooth_flow_scores)

    min_len = min(len(app), len(mot))
    app = app[:min_len]
    mot = mot[:min_len]

    fused = weight_app * app + weight_mot * mot
    return gaussian_filter1d(fused, sigma=sigma)


def detect_anomaly_segments(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    threshold_percentile: float = 95.0,
    window: int = 30
) -> Tuple[List[int], float]:
    """Detect anomalous temporal windows exceeding threshold in a score time-series.

    Args:
        scores: 1D smoothed anomaly score array.
        threshold: Predefined absolute threshold value. If None, percentile is used.
        threshold_percentile: Percentile (e.g. 95) used to calculate adaptive threshold.
        window: Temporal smoothing window size in frames. Defaults to 30.

    Returns:
        Tuple of (detected_start_indices: List[int], threshold_value: float).
    """
    if threshold is None:
        threshold = float(np.percentile(scores, threshold_percentile))

    detected_segments: List[int] = []
    num_frames = len(scores)

    for i in range(num_frames - window):
        window_mean = float(np.mean(scores[i : i + window]))
        if window_mean > threshold:
            detected_segments.append(i)

    return detected_segments, threshold


def temporal_iou(
    pred_segments: List[int],
    gt_start: int,
    gt_end: int,
    window: int = 30
) -> float:
    """Calculate maximum 1D temporal Intersection over Union (IoU) of detection windows against GT.

    Args:
        pred_segments: Starting indices of detected anomaly segments.
        gt_start: Ground truth start frame.
        gt_end: Ground truth end frame.
        window: Segment window width in frames.

    Returns:
        float: Maximum temporal IoU score.
    """
    if not pred_segments:
        return 0.0

    ious: List[float] = []
    for idx in pred_segments:
        pred_start = idx
        pred_end = idx + window

        inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = max(pred_end, gt_end) - min(pred_start, gt_start)

        if union > 0:
            ious.append(inter / union)
        else:
            ious.append(0.0)

    return max(ious) if ious else 0.0


def compute_detection_delay(
    pred_segments: List[int],
    gt_start: int
) -> Optional[int]:
    """Calculate the frame delay between the ground-truth onset and first detected segment.

    Args:
        pred_segments: Detected segment start indices.
        gt_start: Ground truth onset frame.

    Returns:
        Optional[int]: Delay in frames, or None if no detection occurs at or after GT start.
    """
    valid_detections = [idx for idx in pred_segments if idx >= gt_start]
    if len(valid_detections) > 0:
        return valid_detections[0] - gt_start
    return None


def compute_peak_strength(score: np.ndarray) -> float:
    """Compute peak amplitude above the baseline mean score.

    Args:
        score: 1D anomaly score array.

    Returns:
        float: Difference between max score and mean score.
    """
    return float(np.max(score) - np.mean(score))


def evaluate_detection_metrics(
    scores: np.ndarray,
    pred_segments: List[int],
    gt_start: int,
    gt_end: int,
    window: int = 30
) -> Dict[str, Any]:
    """Evaluate frame-level classification and temporal detection performance.

    Computes:
        - True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
        - Precision, Recall, F1-Score
        - Frame-level Intersection over Union (IoU)
        - Temporal Window IoU
        - Detection Delay (frames)
        - Peak Anomaly Strength

    Args:
        scores: 1D anomaly score array.
        pred_segments: Start indices of anomalous windows.
        gt_start: Ground truth anomaly onset frame index.
        gt_end: Ground truth anomaly offset frame index.
        window: Temporal window size.

    Returns:
        Dict[str, Any]: Comprehensive metrics dictionary.
    """
    n_frames = len(scores)
    y_pred = np.zeros(n_frames, dtype=np.int32)
    y_true = np.zeros(n_frames, dtype=np.int32)

    # Mark predicted anomaly frames
    for idx in pred_segments:
        end_idx = min(idx + window, n_frames)
        y_pred[idx:end_idx] = 1

    # Mark ground truth anomaly frames
    clamped_gt_start = max(0, min(gt_start, n_frames))
    clamped_gt_end = max(0, min(gt_end, n_frames))
    y_true[clamped_gt_start:clamped_gt_end] = 1

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    eps = 1e-8
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float(2 * (precision * recall) / (precision + recall + eps))

    intersection = tp
    union = int(np.sum((y_pred == 1) | (y_true == 1)))
    frame_iou = float(intersection / (union + eps))

    tiou = float(temporal_iou(pred_segments, gt_start, gt_end, window=window))
    delay = compute_detection_delay(pred_segments, gt_start)
    peak = compute_peak_strength(scores)

    return {
        "total_frames": n_frames,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "frame_iou": frame_iou,
        "temporal_iou": tiou,
        "detection_delay_frames": delay,
        "peak_strength": peak,
    }
