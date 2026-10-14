# -*- coding: utf-8 -*-
"""
Tạo ảnh kiến trúc so sánh đối đầu giữa SSDLite vs. Faster R-CNN
Xuất ảnh: benchmark_comparison/dual_architecture_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_dual_architectures(out_path="benchmark_comparison/dual_architecture_comparison.png"):
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    c_ssd_bg = "#eafaf1"
    c_ssd_border = "#27ae60"
    c_frcnn_bg = "#ebf5fb"
    c_frcnn_border = "#2980b9"
    c_rec_bg = "#fbfcfc"
    c_rec_border = "#8e44ad"

    # Header
    ax.text(7.5, 8.5, "Comparative Model Pipelines: One-Stage vs. Two-Stage Detection", fontsize=16, fontweight="bold", ha="center")
    ax.text(7.5, 8.1, "Biometric Palm ROI Localization & Downstream Vein Recognition System", fontsize=11, color="#555555", ha="center")

    # ==================== PIPELINE 1: SSDLite-MobileNetV3 (One-Stage) ====================
    ax.add_patch(patches.FancyBboxPatch((0.5, 4.3), 14.0, 3.4, boxstyle="round,pad=0.2", facecolor=c_ssd_bg, edgecolor=c_ssd_border, linewidth=2))
    ax.text(1.0, 7.3, "Model 1: SSDLite320 + MobileNetV3-Large (One-Stage Edge Pipeline)", fontsize=12, fontweight="bold", color=c_ssd_border)

    # Stages for Model 1
    # Step 1.1: Input
    ax.add_patch(patches.FancyBboxPatch((1.0, 4.8), 2.2, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.5))
    ax.text(2.1, 6.3, "Input Image", fontsize=10, fontweight="bold", ha="center")
    ax.text(2.1, 5.5, "RGB / NIR Stream\nResized: 320x320\nNormalized", fontsize=8.5, ha="center", color="#444444")

    ax.annotate("", xy=(3.8, 5.8), xytext=(3.3, 5.8), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Step 1.2: Backbone
    ax.add_patch(patches.FancyBboxPatch((3.8, 4.8), 2.8, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.5))
    ax.text(5.2, 6.3, "MobileNetV3-Large", fontsize=10, fontweight="bold", ha="center")
    ax.text(5.2, 5.5, "Inverted Residuals\nDepthwise Separable Conv\nMulti-scale Feature Maps", fontsize=8.5, ha="center", color="#444444")

    ax.annotate("", xy=(7.2, 5.8), xytext=(6.7, 5.8), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Step 1.3: SSDLite Head
    ax.add_patch(patches.FancyBboxPatch((7.2, 4.8), 3.0, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.5))
    ax.text(8.7, 6.3, "SSDLite Detection Head", fontsize=10, fontweight="bold", ha="center")
    ax.text(8.7, 5.5, "Depthwise Convolutions\nSingle-shot Class & BBox\nDirect NMS Filtering", fontsize=8.5, ha="center", color="#444444")

    ax.annotate("", xy=(10.8, 5.8), xytext=(10.3, 5.8), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Step 1.4: Output
    ax.add_patch(patches.FancyBboxPatch((10.8, 4.8), 3.2, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor=c_ssd_border, linewidth=1.8))
    ax.text(12.4, 6.3, "ROI Output & Metrics", fontsize=10, fontweight="bold", ha="center", color=c_ssd_border)
    ax.text(12.4, 5.4, "• Model Size: ~15 MB\n• Latency: ~20 ms (Real-time)\n• F1-Score: 0.9994 (99.9%)\n• Expand +10% Crop", fontsize=8.5, ha="center", color="#222222")

    # ==================== PIPELINE 2: Faster R-CNN ResNet-50 (Two-Stage) ====================
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 14.0, 3.4, boxstyle="round,pad=0.2", facecolor=c_frcnn_bg, edgecolor=c_frcnn_border, linewidth=2))
    ax.text(1.0, 3.5, "Model 2: Faster R-CNN + ResNet-50-FPN (Two-Stage Server Pipeline)", fontsize=12, fontweight="bold", color=c_frcnn_border)

    # Step 2.1: Input
    ax.add_patch(patches.FancyBboxPatch((1.0, 1.0), 2.2, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.5))
    ax.text(2.1, 2.5, "Input Image", fontsize=10, fontweight="bold", ha="center")
    ax.text(2.1, 1.7, "Adaptive Resolution\nMin: 600px | Max: 1000px\nPyramid Scales", fontsize=8.5, ha="center", color="#444444")

    ax.annotate("", xy=(3.8, 2.0), xytext=(3.3, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Step 2.2: Backbone FPN
    ax.add_patch(patches.FancyBboxPatch((3.8, 1.0), 2.8, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.5))
    ax.text(5.2, 2.5, "ResNet-50 + FPN", fontsize=10, fontweight="bold", ha="center")
    ax.text(5.2, 1.7, "Deep Residual Layers (50)\nFeature Pyramid Network\nMulti-level Receptive Field", fontsize=8.5, ha="center", color="#444444")

    ax.annotate("", xy=(7.2, 2.0), xytext=(6.7, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Step 2.3: Stage 1 RPN + RoIAlign
    ax.add_patch(patches.FancyBboxPatch((7.2, 1.0), 3.0, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.5))
    ax.text(8.7, 2.5, "RPN & RoIAlign Head", fontsize=10, fontweight="bold", ha="center")
    ax.text(8.7, 1.7, "Region Proposal Network\nAnchor Box Generation\nRoIAlign Feature Pooling", fontsize=8.5, ha="center", color="#444444")

    ax.annotate("", xy=(10.8, 2.0), xytext=(10.3, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Step 2.4: Output
    ax.add_patch(patches.FancyBboxPatch((10.8, 1.0), 3.2, 2.0, boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor=c_frcnn_border, linewidth=1.8))
    ax.text(12.4, 2.5, "ROI Output & Metrics", fontsize=10, fontweight="bold", ha="center", color=c_frcnn_border)
    ax.text(12.4, 1.6, "• Model Size: ~165 MB\n• Latency: ~120 ms (Heavy)\n• F1-Score: 0.4545 (Excess proposals)\n• mAP@0.3: 1.0000", fontsize=8.5, ha="center", color="#222222")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved dual architecture diagram to {out_path}")

if __name__ == "__main__":
    draw_dual_architectures()
