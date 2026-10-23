"""Data preprocessing and motion feature extraction utilities for Crowd Anomaly Detection."""

import os
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch


def load_and_preprocess_frame(
    img_path: str,
    resize: Tuple[int, int] = (256, 256)
) -> torch.Tensor:
    """Load an image frame from disk, resize, normalize to [0, 1], and convert to a PyTorch tensor.

    Args:
        img_path: Path to the image file.
        resize: Target resolution (width, height). Defaults to (256, 256).

    Returns:
        torch.Tensor of shape (3, H, W) with float32 values in range [0, 1].

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be decoded.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Frame image not found at: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image at: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, resize)
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return tensor


def compute_farneback_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0
) -> Tuple[np.ndarray, float]:
    """Compute dense optical flow between two consecutive grayscale frames using Farneback's method.

    Args:
        prev_gray: Grayscale image of shape (H, W), dtype uint8.
        curr_gray: Grayscale image of shape (H, W), dtype uint8.
        pyr_scale: Image scale (< 1) to build pyramids for each image.
        levels: Number of pyramid layers including the initial image.
        winsize: Averaging window size.
        iterations: Number of iterations the algorithm does at each pyramid level.
        poly_n: Size of the pixel neighborhood used to find polynomial expansion.
        poly_sigma: Standard deviation of Gaussian used to smooth derivatives.
        flags: Operation flags.

    Returns:
        Tuple of (magnitude_map: np.ndarray, mean_magnitude: float).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(mag))
    return mag, mean_mag


def compute_dataset_optical_flow(dataset) -> np.ndarray:
    """Compute optical flow magnitudes across consecutive frames in a dataset.

    Args:
        dataset: A PyTorch Dataset yielding normalized image tensors of shape (3, H, W).

    Returns:
        numpy.ndarray: Array of mean motion magnitudes for each frame (first element is 0.0).
    """
    flow_magnitudes: List[float] = []
    prev_gray: Optional[np.ndarray] = None

    for i in range(len(dataset)):
        frame = dataset[i]
        if isinstance(frame, torch.Tensor):
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
        else:
            frame_np = frame

        gray = cv2.cvtColor((frame_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        if prev_gray is None:
            prev_gray = gray
            flow_magnitudes.append(0.0)
            continue

        _, mean_mag = compute_farneback_optical_flow(prev_gray, gray)
        flow_magnitudes.append(mean_mag)
        prev_gray = gray

    return np.array(flow_magnitudes, dtype=np.float64)


def standardize_scores(scores: Union[List[float], np.ndarray]) -> np.ndarray:
    """Standardize a score array using Z-score normalization (zero mean, unit variance).

    Args:
        scores: 1D array or list of raw anomaly/motion scores.

    Returns:
        np.ndarray: Z-score standardized array.
    """
    arr = np.asarray(scores, dtype=np.float64)
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def smooth_scores(
    scores: Union[List[float], np.ndarray],
    sigma: float = 10.0
) -> np.ndarray:
    """Apply 1D Gaussian filter smoothing over a temporal sequence of scores.

    Args:
        scores: 1D sequence of scores.
        sigma: Standard deviation for Gaussian kernel. Defaults to 10.0.

    Returns:
        np.ndarray: Smoothed 1D score array.
    """
    arr = np.asarray(scores, dtype=np.float64)
    return gaussian_filter1d(arr, sigma=sigma)
