#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-Only Latency Benchmark for SSDLite-MobileNetV3 Edge Inference Pipeline.
- Runs exclusively on CPU with configurable thread count (default: 4).
- Uses Batch Size = 1.
- Breaks down execution latency into:
    * Preprocessing (letterbox / resize, normalization, tensor prep)
    * Model forward (ONNX Runtime CPUExecutionProvider)
    * Postprocessing (box scaling, confidence thresholding, NMS expansion)
    * Total pipeline latency
- Warms up for N iterations, then profiles for M runs.
- Computes mean, std, p50, p95, and FPS.
- Saves detailed JSON metrics with environment metadata (Python, PyTorch, ORT, git commit, checkpoint).
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any

# Force CPU device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_PATH = REPO_ROOT / "edge_exports" / "ssdlite_mobilenetv3.onnx"
DEFAULT_CHECKPOINT = REPO_ROOT / "models_ssdlite_mobilenetv3" / "output_newdataset" / "checkpoints" / "best_loss.pth"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "edge_exports" / "benchmark_cpu.json"


def get_git_commit(cwd: Path) -> str:
    """Safely obtain current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_cpu_info() -> Dict[str, Any]:
    """Retrieve detailed CPU processor model, core counts, and OS architecture."""
    import platform
    cpu_name = platform.processor() or "unknown"
    if sys.platform == "win32":
        try:
            cmd = "Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, check=True)
            out = res.stdout.strip()
            if out:
                cpu_name = out.splitlines()[0].strip()
        except Exception:
            pass

    return {
        "model": cpu_name,
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "logical_cores": os.cpu_count() or 1,
    }



def preprocess_image(image_bgr: np.ndarray, target_size=(320, 320)) -> np.ndarray:
    """Preprocess raw BGR image to match model input requirements: normalized RGB tensor."""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Resize to (320, 320)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    # Standard normalization: (x / 255.0 - mean) / std
    # In torchvision SSD defaults: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # Input expected is [0, 1] range float tensor, SSD's internal transform standardizes by mean/std
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # (3, H, W)
    tensor = np.expand_dims(tensor, axis=0)   # (1, 3, H, W)
    return tensor


def postprocess_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    orig_shape: tuple,
    conf_thresh: float = 0.3,
    expand_ratio: float = 0.1,
) -> list:
    """Scale bounding boxes back to original coordinates and apply edge expansion."""
    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / 320.0
    scale_y = orig_h / 320.0

    valid_mask = scores >= conf_thresh
    sel_boxes = boxes[valid_mask]
    sel_scores = scores[valid_mask]
    sel_labels = labels[valid_mask] if labels is not None else np.ones(len(sel_boxes), dtype=int)

    results = []
    for box, score, label in zip(sel_boxes, sel_scores, sel_labels):
        x1, y1, x2, y2 = box
        # Scale to original dimensions
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y

        # Bounding box expansion (10% default)
        if expand_ratio > 0:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
            bw_exp, bh_exp = bw * (1.0 + expand_ratio), bh * (1.0 + expand_ratio)
            x1 = max(0, cx - bw_exp / 2.0)
            y1 = max(0, cy - bh_exp / 2.0)
            x2 = min(orig_w, cx + bw_exp / 2.0)
            y2 = min(orig_h, cy + bh_exp / 2.0)

        results.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(score),
            "label": int(label),
        })
    return results


def run_benchmark(
    onnx_path: Path,
    checkpoint_path: Path,
    output_json: Path,
    num_threads: int = 4,
    warmup_runs: int = 20,
    test_runs: int = 100,
    conf_thresh: float = 0.3,
) -> Dict[str, Any]:
    """Execute end-to-end CPU edge inference benchmark."""
    import torch
    import onnxruntime as ort

    print(f"============================================================")
    print(f" Edge Inference CPU Latency Benchmark (SSDLite-MobileNetV3) ")
    print(f"============================================================")
    print(f"[INFO] device=cpu")
    print(f"[INFO] Threads: {num_threads}")
    print(f"[INFO] ONNX Model: {onnx_path}")
    print(f"[INFO] Warmup iterations: {warmup_runs}, Measured iterations: {test_runs}")

    # Set thread count across libraries
    torch.set_num_threads(num_threads)
    cv2.setNumThreads(num_threads)

    # Initialize ONNX Runtime Inference Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Create synthetic test camera frame (e.g. 640x480 standard NIR webcam frame)
    synthetic_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # 1. Warm-up Phase
    print(f"[INFO] Warming up session with {warmup_runs} iterations...")
    for _ in range(warmup_runs):
        inp_tensor = preprocess_image(synthetic_frame)
        outputs = session.run(None, {input_name: inp_tensor})
        _ = postprocess_detections(outputs[0], outputs[1], outputs[2] if len(outputs) > 2 else None, synthetic_frame.shape, conf_thresh)

    # 2. Timing Profile Phase
    print(f"[INFO] Benchmarking {test_runs} iterations...")
    prep_times = []
    forward_times = []
    post_times = []
    total_times = []

    for _ in range(test_runs):
        t_start = time.perf_counter()

        # Preprocessing
        t0 = time.perf_counter()
        inp_tensor = preprocess_image(synthetic_frame)
        t1 = time.perf_counter()

        # Model Forward
        outputs = session.run(None, {input_name: inp_tensor})
        t2 = time.perf_counter()

        # Postprocessing
        _ = postprocess_detections(
            outputs[0],
            outputs[1],
            outputs[2] if len(outputs) > 2 else None,
            synthetic_frame.shape,
            conf_thresh,
        )
        t3 = time.perf_counter()

        t_end = time.perf_counter()

        prep_times.append((t1 - t0) * 1000.0)
        forward_times.append((t2 - t1) * 1000.0)
        post_times.append((t3 - t2) * 1000.0)
        total_times.append((t_end - t_start) * 1000.0)

    # Compute statistics
    def calc_stats(latencies):
        arr = np.array(latencies)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }

    prep_stats = calc_stats(prep_times)
    forward_stats = calc_stats(forward_times)
    post_stats = calc_stats(post_times)
    total_stats = calc_stats(total_times)

    fps = 1000.0 / total_stats["mean_ms"] if total_stats["mean_ms"] > 0 else 0.0

    print("\n------------------- Benchmark Results -------------------")
    print(f"Preprocessing Latency (mean ± std):   {prep_stats['mean_ms']:.2f} ± {prep_stats['std_ms']:.2f} ms")
    print(f"Model Forward Latency (mean ± std):   {forward_stats['mean_ms']:.2f} ± {forward_stats['std_ms']:.2f} ms")
    print(f"Postprocessing Latency (mean ± std):  {post_stats['mean_ms']:.2f} ± {post_stats['std_ms']:.2f} ms")
    print(f"Total Pipeline Latency (mean ± std):  {total_stats['mean_ms']:.2f} ± {total_stats['std_ms']:.2f} ms")
    print(f"Total Latency Percentiles:            p50: {total_stats['p50_ms']:.2f} ms | p95: {total_stats['p95_ms']:.2f} ms")
    print(f"Throughput (FPS):                     {fps:.2f} FPS")
    print("---------------------------------------------------------\n")

    # Collect metadata
    git_hash = get_git_commit(REPO_ROOT)
    cpu_info = get_cpu_info()
    results = {
        "metadata": {
            "device": "cpu",
            "benchmark_environment": "development_workstation",
            "cpu_info": cpu_info,
            "batch_size": 1,
            "cpu_threads": num_threads,
            "warmup_runs": warmup_runs,
            "test_runs": test_runs,
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "git_commit": git_hash,
            "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else "None",
            "onnx_model_path": str(onnx_path.resolve()),
        },
        "metrics": {
            "preprocessing": prep_stats,
            "model_forward": forward_stats,
            "postprocessing": post_stats,
            "total_pipeline": total_stats,
            "throughput_fps": float(fps),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Detailed benchmark results saved to: {output_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark SSDLite ONNX model on CPU for edge devices")
    parser.add_argument("--onnx-path", type=str, default=str(DEFAULT_ONNX_PATH), help="Path to exported .onnx file")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to reference checkpoint")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Output JSON path")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads to use (default: 4)")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations (default: 50)")
    parser.add_argument("--runs", type=int, default=300, help="Number of benchmark iterations (default: 300)")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Detection confidence threshold")

    args = parser.parse_args()

    onnx_p = Path(args.onnx_path).resolve()
    ckpt_p = Path(args.checkpoint).resolve() if args.checkpoint else None
    out_json = Path(args.output_json).resolve()

    if not onnx_p.is_file():
        print(f"[ERROR] ONNX file not found at {onnx_p}. Please run export_ssdlite_edge.py first.", file=sys.stderr)
        sys.exit(1)

    run_benchmark(
        onnx_path=onnx_p,
        checkpoint_path=ckpt_p,
        output_json=out_json,
        num_threads=args.threads,
        warmup_runs=args.warmup,
        test_runs=args.runs,
        conf_thresh=args.conf_thresh,
    )


if __name__ == "__main__":
    main()
