#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export SSDLite-MobileNetV3 checkpoint to ONNX format for edge deployment.
- Runs purely on CPU (device=cpu), strictly avoiding CUDA.
- Reconstructs SSDLite-MobileNetV3 architecture matching the checkpoint weights.
- Supports dynamic batch axes if requested.
- Validates exported ONNX model with onnx.checker.
"""

import os
import sys
import argparse
from pathlib import Path
from functools import partial

# Force CPU device visibility and prevent CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection.ssdlite import (
    _mobilenet_extractor,
    DefaultBoxGenerator,
    SSDLiteHead,
    SSD,
    det_utils,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "models_ssdlite_mobilenetv3" / "output_newdataset" / "checkpoints" / "best_loss.pth"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "edge_exports"


def build_ssdlite_mobilenetv3_cpu(num_classes: int = 2) -> SSD:
    """
    Construct SSDLite320 with MobileNetV3-Large backbone on CPU.
    Uses norm_layer (BatchNorm2d eps=0.001, momentum=0.03) and reduced_tail=False
    to match the trained model feature dimensions exactly (960 channels in C4/C5).
    """
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    backbone = mobilenet_v3_large(
        weights=None,
        norm_layer=norm_layer,
        reduced_tail=False,
    )
    backbone = _mobilenet_extractor(backbone, trainable_layers=6, norm_layer=norm_layer)
    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }

    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **defaults,
    )
    return model


def load_checkpoint_cpu(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    """Load model checkpoint state dict onto CPU."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint on device=cpu from: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))

    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict):
        sd = state
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(state)}")

    cleaned_sd = {}
    for k, v in sd.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        cleaned_sd[new_k] = v

    model.load_state_dict(cleaned_sd)
    model.eval()
    print("[INFO] Model weights loaded successfully onto device=cpu")
    return model


def export_to_onnx(
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    opset_version: int = 14,
    dynamic_batch: bool = True,
    num_classes: int = 2,
) -> Path:
    """Export SSDLite model to ONNX on CPU."""
    device_str = "cpu"
    print(f"[INFO] Initializing export pipeline with device={device_str}")

    try:
        import onnx
    except ImportError:
        raise ImportError("Package 'onnx' is required for export. Please install onnx.")

    ckpt_p = Path(checkpoint_path).resolve()
    out_dir_p = Path(output_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir_p / "ssdlite_mobilenetv3.onnx"

    print(f"[INFO] device={device_str}")
    print(f"[INFO] Building SSDLite-MobileNetV3 model architecture (classes={num_classes})...")
    model = build_ssdlite_mobilenetv3_cpu(num_classes=num_classes)
    model = load_checkpoint_cpu(model, ckpt_p)
    model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 320, 320, dtype=torch.float32, device="cpu")
    input_names = ["images"]
    output_names = ["boxes", "scores", "labels"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
        }
        print("[INFO] Dynamic batch dimension enabled for input 'images'.")

    print(f"[INFO] Exporting ONNX to: {onnx_path} (opset={opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"[INFO] ONNX export completed successfully: {onnx_path}")

    # Check exported ONNX model
    print("[INFO] Checking ONNX model integrity with onnx.checker...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"[INFO] ONNX verification passed! File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export SSDLite-MobileNetV3 Palm ROI detector to ONNX (CPU-only)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to .pth checkpoint")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for ONNX")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    parser.add_argument("--disable-dynamic-batch", action="store_true", help="Disable dynamic batch dimension")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes (default: 2)")

    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        opset_version=args.opset,
        dynamic_batch=not args.disable_dynamic_batch,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
