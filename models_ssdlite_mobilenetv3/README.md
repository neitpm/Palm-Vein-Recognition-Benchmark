# SSDLite-MobileNetV3 Palm ROI & Biometric Recognition Module

This module provides the one-stage edge-optimized detection pipeline and downstream biometric vein recognition.

## Features
- **SSDLite320 with MobileNetV3-Large Backbone**: Ultralight (15 MB), low latency (~20 ms on CPU).
- **Adaptive Boundary Expansion**: Automatically expands bounding box borders by +10% to preserve palm contours.
- **Vein Enhancement & ArcFace Metric Learning**: Integrated CLAHE and PalmVeinNet (EfficientNet-B0 + ArcMarginProduct).

## Training
```bash
python scripts/train_ssd_palm.py --config configs/ssd_palm.yaml
```

## Evaluation
```bash
python scripts/eval_ssd_palm.py --weights output/checkpoints/best_loss.pth --conf 0.30
```

## End-to-End Demo
```bash
python scripts/demo_recognition_pipeline.py
```
