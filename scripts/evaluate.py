#!/usr/bin/env python3
"""CLI script for evaluating Crowd Anomaly Detection models and pipelines."""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CrowdFrameDataset, CrowdSequenceDataset
from src.data.preprocessing import (
    compute_dataset_optical_flow,
    smooth_scores,
    standardize_scores,
)
from src.evaluation.evaluate import (
    compute_hybrid_fusion,
    compute_reconstruction_errors,
    compute_sequence_reconstruction_errors,
    detect_anomaly_segments,
    evaluate_detection_metrics,
)
from src.models.cnn_autoencoder import CrowdAutoencoder
from src.models.convlstm import ConvLSTMAutoencoderV2
from src.training.train import get_device
from src.utils.config import load_config
from src.utils.visualization import (
    plot_anomaly_scores,
    plot_detection_vs_ground_truth,
    plot_model_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Crowd Anomaly Detection pipeline."
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
        help="Path to test frames directory (overrides config)."
    )
    parser.add_argument(
        "--ae_checkpoint",
        type=str,
        default=None,
        help="Path to CNN Autoencoder weights checkpoint (.pt file)."
    )
    parser.add_argument(
        "--convlstm_checkpoint",
        type=str,
        default=None,
        help="Path to ConvLSTM Autoencoder weights checkpoint (.pt file)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'mps', 'cpu', or 'auto')."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save evaluation plots and metrics."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    test_path = args.data_path or config["data"]["test_path"]
    img_size = tuple(config["data"]["image_size"])
    seq_len = config["data"].get("sequence_length", 10)
    device = get_device(args.device)

    results_dir = args.results_dir or config["output"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"==================================================")
    print(f"  Crowd Anomaly Detection Evaluation")
    print(f"  Test Dataset: {test_path}")
    print(f"  Device:       {device}")
    print(f"  Results Dir:  {results_dir}")
    print(f"==================================================")

    if not os.path.exists(test_path):
        print(f"Error: Test directory '{test_path}' does not exist.")
        print("Please verify dataset path or supply --data_path /path/to/test_frames")
        sys.exit(1)

    # 1. Load Datasets
    test_frame_dataset = CrowdFrameDataset(test_path, resize=img_size)
    test_frame_loader = DataLoader(test_frame_dataset, batch_size=1, shuffle=False)

    sigma = config["detection"].get("gaussian_sigma", 10.0)
    window = config["detection"].get("window_size", 30)
    threshold_percentile = config["detection"].get("threshold_percentile", 95.0)
    w_app = config["detection"].get("fusion_weight_app", 0.4)
    w_mot = config["detection"].get("fusion_weight_mot", 0.6)
    gt_start = config["ground_truth"].get("gt_start", 780)
    gt_end = config["ground_truth"].get("gt_end", 830)

    # 2. Appearance Model (CNN Autoencoder)
    print("\n[1/4] Evaluating CNN Autoencoder (Appearance)...")
    ae_model = CrowdAutoencoder(in_channels=config["model_ae"].get("in_channels", 3)).to(device)
    ae_ckpt = args.ae_checkpoint or config["model_ae"].get("checkpoint_path")
    if ae_ckpt and os.path.exists(ae_ckpt):
        ae_model.load_state_dict(torch.load(ae_ckpt, map_location=device))
        print(f"Loaded AE checkpoint from: {ae_ckpt}")
    else:
        print("Notice: No AE checkpoint provided or found. Using initialized model.")

    ae_raw_errors = compute_reconstruction_errors(ae_model, test_frame_loader, device=device)
    ae_z = standardize_scores(ae_raw_errors)
    ae_smooth = smooth_scores(ae_z, sigma=sigma)

    plot_anomaly_scores(
        ae_smooth,
        threshold=1.5,
        title="Crowd Autoencoder (Appearance) Anomaly Score",
        save_path=os.path.join(results_dir, "ae_anomaly_score.png")
    )

    # 3. Motion Model (Farneback Optical Flow)
    print("\n[2/4] Computing Optical Flow Motion Magnitudes...")
    flow_mags = compute_dataset_optical_flow(test_frame_dataset)
    flow_z = standardize_scores(flow_mags)
    flow_smooth = smooth_scores(flow_z, sigma=sigma)

    motion_segments, motion_threshold = detect_anomaly_segments(
        flow_smooth,
        threshold_percentile=threshold_percentile,
        window=window
    )

    plot_detection_vs_ground_truth(
        flow_smooth,
        threshold=motion_threshold,
        pred_segments=motion_segments,
        gt_start=gt_start,
        gt_end=gt_end,
        window=window,
        title="Optical Flow Crowd Motion Anomaly Detection",
        save_path=os.path.join(results_dir, "optical_flow_detection.png")
    )

    # 4. Hybrid Fusion Model (AE + Optical Flow)
    print("\n[3/4] Computing Hybrid Spatial + Motion Fusion...")
    final_score = compute_hybrid_fusion(
        smooth_ae_scores=ae_smooth,
        smooth_flow_scores=flow_smooth,
        weight_app=w_app,
        weight_mot=w_mot,
        sigma=sigma
    )

    final_segments, final_threshold = detect_anomaly_segments(
        final_score,
        threshold_percentile=threshold_percentile,
        window=window
    )

    metrics_fusion = evaluate_detection_metrics(
        scores=final_score,
        pred_segments=final_segments,
        gt_start=gt_start,
        gt_end=gt_end,
        window=window
    )

    plot_detection_vs_ground_truth(
        final_score,
        threshold=final_threshold,
        pred_segments=final_segments,
        gt_start=gt_start,
        gt_end=gt_end,
        window=window,
        title="Final Hybrid Fusion Detection vs Ground Truth",
        save_path=os.path.join(results_dir, "hybrid_fusion_detection.png")
    )

    # 5. Temporal Model (ConvLSTM) if sequence dataset available
    lstm_smooth = None
    if len(test_frame_dataset) > seq_len:
        print("\n[4/4] Evaluating Spatio-Temporal ConvLSTM Autoencoder...")
        test_seq_dataset = CrowdSequenceDataset(test_path, seq_len=seq_len, resize=img_size)
        test_seq_loader = DataLoader(test_seq_dataset, batch_size=1, shuffle=False)

        cfg_lstm = config.get("model_convlstm", {})
        lstm_model = ConvLSTMAutoencoderV2(
            in_channels=cfg_lstm.get("in_channels", 3),
            hidden_dim=cfg_lstm.get("hidden_dim", 64),
            kernel_size=cfg_lstm.get("kernel_size", 3)
        ).to(device)

        lstm_ckpt = args.convlstm_checkpoint or cfg_lstm.get("checkpoint_path")
        if lstm_ckpt and os.path.exists(lstm_ckpt):
            lstm_model.load_state_dict(torch.load(lstm_ckpt, map_location=device))
            print(f"Loaded ConvLSTM checkpoint from: {lstm_ckpt}")
        else:
            print("Notice: No ConvLSTM checkpoint provided or found. Using initialized model.")

        lstm_raw_errors = compute_sequence_reconstruction_errors(lstm_model, test_seq_loader, device=device)
        lstm_z = standardize_scores(lstm_raw_errors)
        lstm_smooth = smooth_scores(lstm_z, sigma=sigma)

        plot_anomaly_scores(
            lstm_smooth,
            title="ConvLSTM Spatio-Temporal Anomaly Score",
            save_path=os.path.join(results_dir, "convlstm_anomaly_score.png")
        )

    # 6. Model Comparison Alignment
    print("\n--- Model Benchmark Summary ---")
    aligned_scores = {
        "AE (Appearance)": ae_smooth[seq_len:],
        "Flow (Motion)": flow_smooth[seq_len:],
        "AE + Flow (Fusion)": final_score[seq_len:],
    }
    if lstm_smooth is not None:
        aligned_scores["ConvLSTM"] = lstm_smooth

    min_len = min(len(s) for s in aligned_scores.values())
    aligned_scores = {k: v[:min_len] for k, v in aligned_scores.items()}
    aligned_gt = max(0, gt_start - seq_len)

    plot_model_comparison(
        aligned_scores_dict=aligned_scores,
        aligned_gt=aligned_gt,
        save_path=os.path.join(results_dir, "model_comparison.png")
    )

    print("\n================ Hybrid Fusion Results ================")
    print(f"Precision:        {metrics_fusion['precision']:.4f}")
    print(f"Recall:           {metrics_fusion['recall']:.4f}")
    print(f"F1 Score:         {metrics_fusion['f1_score']:.4f}")
    print(f"Frame IoU:        {metrics_fusion['frame_iou']:.4f}")
    print(f"Temporal IoU:     {metrics_fusion['temporal_iou']:.4f}")
    print(f"Detection Delay:  {metrics_fusion['detection_delay_frames']} frames")
    print(f"Peak Strength:    {metrics_fusion['peak_strength']:.4f}")
    print("=======================================================")

    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_fusion, f, indent=2)
    print(f"\nSaved evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
