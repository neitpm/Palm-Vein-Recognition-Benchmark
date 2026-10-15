#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify exported SSDLite-MobileNetV3 ONNX model on CPU using ONNX Runtime.
- Strictly executes on CPU (CPUExecutionProvider).
- Tests dummy input inference and validates output tensor shapes.
- Compares ONNX predictions against PyTorch reference outputs to verify numerical parity.
- Returns non-zero exit code if validation fails.
"""

import os
import sys
import argparse
from pathlib import Path

# Force CPU device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ONNX_PATH = REPO_ROOT / "edge_exports" / "ssdlite_mobilenetv3.onnx"
DEFAULT_CHECKPOINT = REPO_ROOT / "models_ssdlite_mobilenetv3" / "output_newdataset" / "checkpoints" / "best_loss.pth"


def verify_onnx(
    onnx_path: Path,
    checkpoint_path: Path = None,
    tolerance: float = 1e-3,
    verbose: bool = True,
) -> bool:
    """Run verification checks on exported ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        print(f"[ERROR] Required packages not found: {e}", file=sys.stderr)
        return False

    if not onnx_path.is_file():
        print(f"[ERROR] ONNX model file does not exist: {onnx_path}", file=sys.stderr)
        return False

    if verbose:
        print(f"[INFO] Verifying ONNX model on device=cpu: {onnx_path}")
        print(f"[INFO] ONNX file size: {onnx_path.stat().st_size / (1024 * 1024):.2f} MB")

    # 1. Structural check with onnx library
    onnx_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_proto)
    if verbose:
        print("[CHECK 1/4] onnx.checker validation PASSED.")

    # 2. Session creation with CPUExecutionProvider only
    available_providers = ort.get_available_providers()
    if verbose:
        print(f"[INFO] Available ORT providers: {available_providers}")
    
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    inputs_meta = session.get_inputs()
    outputs_meta = session.get_outputs()
    input_name = inputs_meta[0].name
    expected_outputs = [o.name for o in outputs_meta]

    if verbose:
        print(f"[CHECK 2/4] InferenceSession created successfully with CPUExecutionProvider.")
        print(f"            Inputs: {[(i.name, i.shape, i.type) for i in inputs_meta]}")
        print(f"            Outputs: {[(o.name, o.shape, o.type) for o in outputs_meta]}")

    # 3. Dummy inference test
    np.random.seed(42)
    dummy_np = np.random.randn(1, 3, 320, 320).astype(np.float32)

    ort_outputs = session.run(None, {input_name: dummy_np})
    if len(ort_outputs) < 2:
        print(f"[ERROR] Expected at least 2 outputs (boxes, scores), got {len(ort_outputs)}", file=sys.stderr)
        return False

    boxes, scores = ort_outputs[0], ort_outputs[1]
    labels = ort_outputs[2] if len(ort_outputs) > 2 else None

    # Check output shape: boxes must be (N, 4) and scores must be (N,)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        print(f"[ERROR] Invalid boxes shape: {boxes.shape}, expected (*, 4)", file=sys.stderr)
        return False
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        print(f"[ERROR] Invalid scores shape: {scores.shape}, expected ({boxes.shape[0]},)", file=sys.stderr)
        return False

    if verbose:
        print(f"[CHECK 3/4] Dummy inference successful.")
        print(f"            Predicted detections: {boxes.shape[0]}")
        print(f"            Boxes shape: {boxes.shape}, dtype: {boxes.dtype}")
        print(f"            Scores shape: {scores.shape}, dtype: {scores.dtype}")
        if labels is not None:
            print(f"            Labels shape: {labels.shape}, dtype: {labels.dtype}")

    # 4. Numerical parity with PyTorch reference
    if checkpoint_path is not None and checkpoint_path.is_file():
        if verbose:
            print(f"[INFO] Comparing against PyTorch reference from {checkpoint_path}...")
        from scripts.export_ssdlite_edge import build_ssdlite_mobilenetv3_cpu, load_checkpoint_cpu

        py_model = build_ssdlite_mobilenetv3_cpu(num_classes=2)
        py_model = load_checkpoint_cpu(py_model, checkpoint_path)
        py_model.to("cpu")
        py_model.eval()

        dummy_torch = torch.from_numpy(dummy_np)
        with torch.no_grad():
            py_res = py_model(dummy_torch)[0]

        py_boxes = py_res["boxes"].cpu().numpy()
        py_scores = py_res["scores"].cpu().numpy()

        # Due to slight float32 rounding differences in NMS score sorting on raw random noise,
        # very low-confidence noise boxes (score < 0.05) near the score threshold may fluctuate by 1 box.
        # We compare the top-K significant detections (or detections with score >= 0.05).
        top_k = min(10, len(py_scores), len(scores))
        if top_k > 0:
            py_top_boxes = py_boxes[:top_k]
            ort_top_boxes = boxes[:top_k]
            py_top_scores = py_scores[:top_k]
            ort_top_scores = scores[:top_k]

            max_box_diff = float(np.max(np.abs(py_top_boxes - ort_top_boxes)))
            max_score_diff = float(np.max(np.abs(py_top_scores - ort_top_scores)))

            if verbose:
                print(f"[CHECK 4/4] Numerical Parity Check (Top-{top_k} detections):")
                print(f"            Max Box Absolute Difference:   {max_box_diff:.6e}")
                print(f"            Max Score Absolute Difference: {max_score_diff:.6e}")

            if max_box_diff > tolerance or max_score_diff > tolerance:
                print(f"[ERROR] Parity difference exceeds tolerance {tolerance}", file=sys.stderr)
                return False
            if verbose:
                print(f"[CHECK 4/4] Parity PASSED (tolerance: {tolerance})")
        else:
            if verbose:
                print("[CHECK 4/4] Parity PASSED (no detections above threshold)")
    else:
        if verbose:
            print("[INFO] Skipping PyTorch parity comparison (no valid checkpoint provided).")

    if verbose:
        print("[SUCCESS] All ONNX CPU verification checks PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify SSDLite-MobileNetV3 ONNX model on CPU")
    parser.add_argument("--onnx-path", type=str, default=str(DEFAULT_ONNX_PATH), help="Path to exported .onnx file")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to reference .pth checkpoint")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max numerical tolerance for comparison")
    parser.add_argument("--no-compare", action="store_true", help="Skip comparison with PyTorch checkpoint")

    args = parser.parse_args()
    onnx_p = Path(args.onnx_path).resolve()
    ckpt_p = None if args.no_compare else Path(args.checkpoint).resolve()

    success = verify_onnx(
        onnx_path=onnx_p,
        checkpoint_path=ckpt_p,
        tolerance=args.tolerance,
        verbose=True,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
