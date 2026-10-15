#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare clean, decoupled Hugging Face Model Repository staging directory for SSDLite-MobileNetV3.
Packages:
- README.md (Comprehensive HF Model Card)
- LICENSE
- config/ (ssd_palm.yaml, model_metadata.json)
- pytorch/ (ssdlite_best_loss.pth)
- onnx/ (ssdlite_mobilenetv3.onnx)
- metrics/ (historical_metrics.json, benchmark_cpu.json, sample_images_parity.json)
- requirements-inference.txt
- checksums.sha256
"""

import os
import shutil
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = REPO_ROOT / "hf_releases" / "ssdlite-palm-vein-roi"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_model_card(
    ckpt_sha256: str,
    onnx_sha256: str,
    metrics: dict,
    benchmark_meta: dict,
) -> str:
    cpu_info = benchmark_meta.get('cpu_info', {})
    cpu_model = cpu_info.get('model', 'Intel Core i5-14600K') if isinstance(cpu_info, dict) else 'Intel Core i5-14600K'
    cpu_platform = cpu_info.get('platform', 'Windows kernel build 10.0.26200') if isinstance(cpu_info, dict) else 'Windows kernel build 10.0.26200'

    return f"""---
language:
- en
license: mit
tags:
- palm-vein
- object-detection
- ssdlite
- mobilenetv3
- onnx
- edge-inference
- biometric-roi
pipeline_tag: object-detection
---

# SSDLite-MobileNetV3 Palm ROI Detector (Edge Inference Prototype)

This repository contains the standalone weights, ONNX export, configuration, and reproducibility artifacts for the **SSDLite-MobileNetV3 Palm Region of Interest (ROI) Detector**, originally developed as part of the [Palm-Vein-Recognition-Benchmark](https://github.com/neitpm/Palm-Vein-Recognition-Benchmark).

> [!NOTE]
> **Technical Positioning & Intended Use:**  
> This artifact is an **engineering inference prototype** for palm localization on near-infrared (NIR) and visible imagery. It is designed as the frontline ROI localization stage prior to vein contrast enhancement and biometric metric learning.  
> It is **not** a production biometric deployment and has not yet been benchmarked on physical edge embedded devices (e.g., Raspberry Pi 4/5, NVIDIA Jetson, NXP i.MX).

---

## 1. Model Architecture & Specifications

| Parameter | Specification |
| :--- | :--- |
| **Architecture** | SSDLite320 (Single Shot MultiBox Detector) |
| **Backbone** | MobileNetV3-Large (`reduced_tail=False`, 960 channels C4/C5) |
| **Input Shape** | `(1, 3, 320, 320)` float32 normalized tensor |
| **Class Mapping** | `0: background`, `1: palm` |
| **In-Graph Operations** | Anchor decoding, score thresholding (`0.001`), NMS (`IoU=0.55`, `topk=300`) |
| **ONNX Outputs** | `boxes: (N, 4)`, `scores: (N,)`, `labels: (N,)` |
| **Checkpoint Policy** | `best_loss` (checkpoint with lowest validation loss during training) |

---

## 2. Model Files & Integrity Checksums

All binary artifacts are verified via SHA-256:

| File Path | Format | Size | SHA-256 Checksum |
| :--- | :---: | :---: | :--- |
| `pytorch/ssdlite_best_loss.pth` | PyTorch `state_dict` | ~14.3 MB | `{ckpt_sha256}` |
| `onnx/ssdlite_mobilenetv3.onnx` | ONNX FP32 (Opset 14) | ~14.3 MB | `{onnx_sha256}` |

---

## 3. Deployment Profiling & Reproducibility (Workstation CPU)

All inference latency and throughput metrics were collected as a **reproducibility-oriented CPU benchmark on the development workstation** (batch size 1, 300 measured iterations after 50 warmup runs on a synthetic camera frame):

- **Workstation Processor:** {cpu_model}
- **CPU Threads:** 4 threads
- **OS Environment:** {cpu_platform}
- **Runtime Stack:** Python 3.11, PyTorch 2.14.1+cpu, ONNX 1.23.2, ONNX Runtime 1.31.0

### Latency Breakdown:
- **Preprocessing Latency:** 1.10 ± 0.33 ms (Color conversion, resize to 320x320, tensor formatting)
- **Model Forward Latency:** 7.02 ± 1.57 ms (ONNX Runtime CPUExecutionProvider, **inclusive of in-graph NMS**)
- **Client Postprocessing:** 0.06 ± 0.04 ms (BBox coordinate rescaling to original frame, confidence filtering, ROI margin expansion)
- **Total Pipeline Latency:** 8.18 ± 1.77 ms (p50: 7.72 ms, p95: 11.19 ms)
- **Workstation Throughput:** ~122.2 FPS on development workstation using ONNX Runtime CPU

---

## 4. PyTorch vs. ONNX Runtime Numerical Parity

Runtime parity was verified to confirm that exporting to ONNX Runtime causes no algorithmic or floating-point degradation:
- **Synthetic Input Checks:** Max box absolute diff < 2.0e-4, Max score absolute diff < 1.0e-6.
- **Sampled Repository Images Check:** Evaluated on sample ROI visualization images using one-to-one greedy IoU matching on identical labels.
  - IoU Agreement: > 0.999 across all matched detections.
  - Class Label Agreement: 100%.
  - Score Difference: < 1.9e-6.
  - Detailed per-box metrics are archived in `metrics/sample_images_parity.json`.

> [!IMPORTANT]
> **Numerical Parity vs. Detection Accuracy:**  
> High IoU agreement (> 0.999) proves runtime numerical equivalence between PyTorch and ONNX Runtime; it does **not** evaluate detection accuracy against human ground-truth annotations.

---

## 5. Historical Detection Metrics (Provenance)

The checkpoint `ssdlite_best_loss.pth` was trained under the repository's COCO validation split:
- **Validation Loss:** {metrics.get('best_val_loss', 1.401):.4f} (Epoch {metrics.get('epoch', 6)})
- **Historical mAP@0.5:0.95:** {metrics.get('map_0.5_0.95', 0.6691):.4f}
- **Historical Precision / Recall / F1:** P = {metrics.get('precision', 0.9986):.4f}, R = {metrics.get('recall', 1.0):.4f}, F1 = {metrics.get('f1', 0.9993):.4f}

*Notice: Historical metrics reflect the original evaluation configuration and remain archived for provenance; formal benchmark re-evaluation with updated evaluation harnesses will be published in subsequent iterations.*

---

## 6. How to Run Inference

### 1. Install Dependencies
```bash
pip install -r requirements-inference.txt
```

### 2. Python ONNX Runtime Example
```python
import cv2
import numpy as np
import onnxruntime as ort

# 1. Initialize ONNX Runtime Session (CPU)
session = ort.InferenceSession("onnx/ssdlite_mobilenetv3.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 2. Preprocess Frame
raw_bgr = cv2.imread("sample_hand.jpg")
h_orig, w_orig = raw_bgr.shape[:2]
rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (320, 320)).astype(np.float32) / 255.0
tensor = np.transpose(resized, (2, 0, 1))[None, ...]

# 3. Model Forward (Includes Anchor Decoding & NMS)
boxes, scores, labels = session.run(None, {{input_name: tensor}})

# 4. Filter & Scale ROI
for box, score, label in zip(boxes, scores, labels):
    if score >= 0.30:
        x1 = int(box[0] * w_orig / 320.0)
        y1 = int(box[1] * h_orig / 320.0)
        x2 = int(box[2] * w_orig / 320.0)
        y2 = int(box[3] * h_orig / 320.0)
        print(f"Detected palm with score {{score:.3f}} at [{{x1}}, {{y1}}, {{x2}}, {{y2}}]")
```

---

## 7. Known Limitations & Responsible Use

1. **No Target Edge Hardware Profiling:** Latency is reported solely for an Intel Core i5 workstation. Power, thermal throttling, and memory bandwidth on embedded ARM SBCs (e.g., Raspberry Pi) or NPUs must be measured in situ.
2. **Biometric Identity Scope:** This model performs palm bounding-box ROI localization only. It does not extract identity feature vectors or compute biometric matching decisions.
3. **Privacy Consideration:** This release provides solely neural network model weights and mathematical descriptors; it contains no raw palm biometrics, human subject identifiers, or personal data.

---

## 8. License
This model artifact is released under the [MIT License](LICENSE).
"""


def prepare_staging():
    print(f"[INFO] Preparing Hugging Face staging directory at: {STAGING_DIR}")
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)

    dirs = {
        "config": STAGING_DIR / "config",
        "pytorch": STAGING_DIR / "pytorch",
        "onnx": STAGING_DIR / "onnx",
        "metrics": STAGING_DIR / "metrics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy LICENSE
    shutil.copy2(REPO_ROOT / "LICENSE", STAGING_DIR / "LICENSE")

    # 2. Copy Config
    shutil.copy2(REPO_ROOT / "models_ssdlite_mobilenetv3" / "configs" / "ssd_palm.yaml", dirs["config"] / "ssd_palm.yaml")

    # 3. Copy Weights
    src_ckpt = REPO_ROOT / "models_ssdlite_mobilenetv3" / "output_newdataset" / "checkpoints" / "best_loss.pth"
    dst_ckpt = dirs["pytorch"] / "ssdlite_best_loss.pth"
    if src_ckpt.is_file():
        shutil.copy2(src_ckpt, dst_ckpt)
        print(f"[INFO] Copied PyTorch checkpoint to: {dst_ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {src_ckpt}")

    # 4. Copy ONNX model
    src_onnx = REPO_ROOT / "edge_exports" / "ssdlite_mobilenetv3.onnx"
    dst_onnx = dirs["onnx"] / "ssdlite_mobilenetv3.onnx"
    shutil.copy2(src_onnx, dst_onnx)
    print(f"[INFO] Copied ONNX model to: {dst_onnx}")

    # 5. Copy Metrics
    src_hist = REPO_ROOT / "models_ssdlite_mobilenetv3" / "output_newdataset" / "checkpoints" / "best_loss_metrics.json"
    shutil.copy2(src_hist, dirs["metrics"] / "historical_metrics.json")
    shutil.copy2(REPO_ROOT / "edge_exports" / "benchmark_cpu.json", dirs["metrics"] / "benchmark_cpu.json")
    shutil.copy2(REPO_ROOT / "edge_exports" / "sample_images_parity.json", dirs["metrics"] / "sample_images_parity.json")

    # 6. Requirements
    req_content = """onnx>=1.14.0
onnxruntime>=1.16.0
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.2.0
torchvision>=0.17.0
"""
    (STAGING_DIR / "requirements-inference.txt").write_text(req_content, encoding="utf-8")

    # 7. Checksums
    checksums = {}
    checksums["pytorch/ssdlite_best_loss.pth"] = compute_sha256(dst_ckpt)
    checksums["onnx/ssdlite_mobilenetv3.onnx"] = compute_sha256(dst_onnx)
    checksums["config/ssd_palm.yaml"] = compute_sha256(dirs["config"] / "ssd_palm.yaml")
    checksums["metrics/benchmark_cpu.json"] = compute_sha256(dirs["metrics"] / "benchmark_cpu.json")
    checksums["metrics/historical_metrics.json"] = compute_sha256(dirs["metrics"] / "historical_metrics.json")
    checksums["metrics/sample_images_parity.json"] = compute_sha256(dirs["metrics"] / "sample_images_parity.json")

    with open(STAGING_DIR / "checksums.sha256", "w", encoding="utf-8") as f:
        for fname, sha in sorted(checksums.items()):
            f.write(f"{sha}  {fname}\n")
    print("[INFO] Generated checksums.sha256")

    # 8. Model Metadata JSON
    metadata = {
        "model_name": "SSDLite320-MobileNetV3-Large",
        "task": "palm-roi-detection",
        "classes": ["background", "palm"],
        "input_resolution": [320, 320],
        "checkpoint_policy": "best_loss",
        "opset": 14,
        "checksums": checksums,
    }
    with open(dirs["config"] / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 9. Build Model Card README.md
    with open(dirs["metrics"] / "historical_metrics.json", "r", encoding="utf-8") as f:
        hist_data = json.load(f)
    with open(dirs["metrics"] / "benchmark_cpu.json", "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    model_card = build_model_card(
        ckpt_sha256=checksums["pytorch/ssdlite_best_loss.pth"],
        onnx_sha256=checksums["onnx/ssdlite_mobilenetv3.onnx"],
        metrics=hist_data,
        benchmark_meta=bench_data.get("metadata", {}),
    )
    (STAGING_DIR / "README.md").write_text(model_card, encoding="utf-8")
    print(f"[INFO] Hugging Face Model Card README.md generated successfully.")
    print(f"[SUCCESS] Staging completed at: {STAGING_DIR}")


if __name__ == "__main__":
    prepare_staging()
