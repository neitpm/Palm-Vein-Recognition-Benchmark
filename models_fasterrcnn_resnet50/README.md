# Faster R-CNN Palm ROI Detection Module

This module implements two-stage deep object detection using `Faster R-CNN` paired with `ResNet-50-FPN` and `EfficientNet-B0` backbones.

## Features
- **Two-Stage Detection**: RPN (Region Proposal Network) + RoIAlign for multi-scale feature extraction.
- **Configurable Resolutions**: Supports multi-scale training (`min_size: 600`, `max_size: 1000`).
- **Comprehensive Diagnostics**: Diagnostic scripts for analyzing anchor boxes and IoU distributions.

## Training
```bash
python scripts/train_fasterrcnn_resnet50_palm.py --config configs/fasterrcnn_resnet50_palm.yaml
```

## Evaluation
```bash
python scripts/eval_fasterrcnnr_resnet50_palm.py
```
