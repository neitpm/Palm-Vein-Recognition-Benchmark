#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CPU-only SSDLite-MobileNetV3 Edge Export and Inference Pipeline.
Tests verify:
1. SSDLite checkpoint exists.
2. Export module is importable and functional on CPU.
3. ONNX model can be exported without errors.
4. ONNX model can be loaded by ONNX Runtime using CPUExecutionProvider.
5. Batch size = 1 inference succeeds and yields valid detection shapes.
6. CUDA is strictly not required and no CUDA calls are invoked.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

# Force CPU device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_PATH = REPO_ROOT / "models_ssdlite_mobilenetv3" / "output_newdataset" / "checkpoints" / "best_loss.pth"


def test_checkpoint_exists():
    """Verify that the target SSDLite checkpoint exists on disk if training artifacts are downloaded."""
    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Training checkpoint (.pth) is excluded by .gitignore and not downloaded: {CHECKPOINT_PATH}")
    assert CHECKPOINT_PATH.is_file()


def test_export_script_importable():
    """Verify export module can be imported without error and exposes required helpers."""
    from scripts.export_ssdlite_edge import (
        build_ssdlite_mobilenetv3_cpu,
        load_checkpoint_cpu,
        export_to_onnx,
    )
    assert callable(build_ssdlite_mobilenetv3_cpu)
    assert callable(load_checkpoint_cpu)
    assert callable(export_to_onnx)


def test_export_onnx_cpu_pipeline():
    """Verify export to ONNX runs purely on CPU and creates valid ONNX file."""
    if not CHECKPOINT_PATH.is_file():
        pytest.skip("Checkpoint file is required for re-export test")

    from scripts.export_ssdlite_edge import export_to_onnx

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_export_dir = Path(tmp_dir) / "edge_exports"
        onnx_file = export_to_onnx(
            checkpoint_path=str(CHECKPOINT_PATH),
            output_dir=str(tmp_export_dir),
            opset_version=14,
            dynamic_batch=True,
            num_classes=2,
        )

        assert onnx_file.is_file(), f"Exported ONNX file was not found at {onnx_file}"
        assert onnx_file.stat().st_size > 1_000_000, "Exported ONNX file appears too small or corrupt"


def test_onnx_load_cpu_and_inference_batch1():
    """Verify exported ONNX model loads with CPUExecutionProvider and runs inference on batch size 1."""
    import onnxruntime as ort

    onnx_file = REPO_ROOT / "edge_exports" / "ssdlite_mobilenetv3.onnx"
    if not onnx_file.is_file():
        if not CHECKPOINT_PATH.is_file():
            pytest.skip("Neither tracked ONNX artifact nor .pth checkpoint found")
        from scripts.export_ssdlite_edge import export_to_onnx
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_export_dir = Path(tmp_dir) / "edge_exports"
            onnx_file = export_to_onnx(
                checkpoint_path=str(CHECKPOINT_PATH),
                output_dir=str(tmp_export_dir),
                opset_version=14,
                dynamic_batch=True,
                num_classes=2,
            )

    # Verification routine check
    from scripts.verify_onnx_cpu import verify_onnx
    passed = verify_onnx(
        onnx_path=onnx_file,
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else None,
        tolerance=1e-3,
        verbose=False,
    )
    assert passed is True, "verify_onnx failed on exported model"

    # Direct ONNX Runtime test
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(onnx_file),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})

    assert len(outputs) >= 2, f"Expected at least boxes and scores, got {len(outputs)}"
    boxes, scores = outputs[0], outputs[1]

    assert boxes.ndim == 2 and boxes.shape[1] == 4, f"Invalid boxes shape: {boxes.shape}"
    assert scores.ndim == 1, f"Invalid scores shape: {scores.shape}"
    assert len(boxes) == len(scores), "Mismatch between box count and score count"


def test_cpu_execution_provider_and_no_cuda_requirement():
    """Verify that ONNX Runtime operates with CPUExecutionProvider and export script enforces CPU."""
    import onnxruntime as ort
    from scripts.export_ssdlite_edge import build_ssdlite_mobilenetv3_cpu

    # CPUExecutionProvider must always be available
    assert "CPUExecutionProvider" in ort.get_available_providers(), "CPUExecutionProvider must be available"

    # Export builder model must be on CPU device without calling CUDA
    model = build_ssdlite_mobilenetv3_cpu(num_classes=2)
    first_param = next(model.parameters())
    assert first_param.device.type == "cpu", f"Model parameters must reside on cpu, found: {first_param.device}"


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def test_sampled_repository_images_numerical_parity():
    """
    Verify numerical parity between PyTorch and ONNX predictions on sampled repository images:
    - Verifies exact detection count match post-filtering.
    - Matches detections via one-to-one greedy IoU matching on identical class labels.
    - Checks that every PyTorch detection finds a matching ONNX counterpart with IoU > 0.999.
    - Validates label exact match and score difference < 1e-4.
    - Clarification: This validates NUMERICAL PARITY between runtimes, not ground-truth detection accuracy.
    """
    import cv2
    import onnxruntime as ort
    from scripts.export_ssdlite_edge import build_ssdlite_mobilenetv3_cpu, load_checkpoint_cpu
    from scripts.benchmark_edge_cpu import postprocess_detections

    sample_dir = REPO_ROOT / "models_ssdlite_mobilenetv3" / "runs" / "rois"
    sample_images = list(sample_dir.glob("*_vis.jpg"))
    if not sample_images:
        pytest.skip(f"No sample palm images found in {sample_dir}")

    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Checkpoint file (.pth) required for PyTorch parity comparison: {CHECKPOINT_PATH}")

    # Load PyTorch model
    py_model = build_ssdlite_mobilenetv3_cpu(num_classes=2)
    py_model = load_checkpoint_cpu(py_model, CHECKPOINT_PATH)
    py_model.eval()

    # Load ONNX Runtime session
    onnx_file = REPO_ROOT / "edge_exports" / "ssdlite_mobilenetv3.onnx"
    if not onnx_file.is_file():
        from scripts.export_ssdlite_edge import export_to_onnx
        onnx_file = export_to_onnx(
            checkpoint_path=str(CHECKPOINT_PATH),
            output_dir=str(REPO_ROOT / "edge_exports"),
            opset_version=14,
            dynamic_batch=True,
            num_classes=2,
        )

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(onnx_file),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    for img_path in sample_images[:6]:
        bgr = cv2.imread(str(img_path))
        assert bgr is not None, f"Failed to read image {img_path}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_320 = cv2.resize(rgb, (320, 320)).astype(np.float32) / 255.0
        t = np.transpose(rgb_320, (2, 0, 1))[None, ...]

        # PyTorch inference
        with torch.no_grad():
            py_res = py_model(torch.from_numpy(t))[0]

        # ONNX inference
        ort_res = session.run(None, {input_name: t})

        # Postprocessing with confidence threshold = 0.3 and expansion = 10%
        py_dets = postprocess_detections(
            py_res["boxes"].cpu().numpy(),
            py_res["scores"].cpu().numpy(),
            py_res["labels"].cpu().numpy(),
            bgr.shape,
            conf_thresh=0.3,
            expand_ratio=0.10,
        )
        ort_dets = postprocess_detections(
            ort_res[0],
            ort_res[1],
            ort_res[2],
            bgr.shape,
            conf_thresh=0.3,
            expand_ratio=0.10,
        )

        assert len(py_dets) == len(ort_dets), (
            f"Detection count mismatch on {img_path.name}: PyTorch={len(py_dets)}, ORT={len(ort_dets)}"
        )

        matched_ort_indices = set()
        for py_d in py_dets:
            best_iou = -1.0
            best_idx = -1
            for j, ort_d in enumerate(ort_dets):
                if j in matched_ort_indices:
                    continue
                if ort_d["label"] != py_d["label"]:
                    continue
                iou = compute_iou(py_d["box"], ort_d["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            assert best_idx != -1, f"No matching ONNX detection found for PyTorch box on {img_path.name}"
            matched_ort_indices.add(best_idx)
            matched_ort_d = ort_dets[best_idx]

            assert best_iou > 0.999, f"IoU between PyTorch and ONNX is too low: {best_iou:.6f} on {img_path.name}"
            assert abs(py_d["score"] - matched_ort_d["score"]) < 1e-4, (
                f"Score difference too high: {abs(py_d['score'] - matched_ort_d['score'])} on {img_path.name}"
            )

        assert len(matched_ort_indices) == len(ort_dets), "Not all ONNX detections were uniquely matched"

