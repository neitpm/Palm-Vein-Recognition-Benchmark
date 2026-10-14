# -*- coding: utf-8 -*-
"""
Script vẽ sơ đồ kiến trúc chi tiết (Model & Pipeline Architecture) dạng đồ họa chất lượng cao.
Xuất ảnh: docs/images/pipeline_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_pipeline_diagram(out_path="docs/images/pipeline_architecture.png"):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Màu sắc chủ đạo (Modern Tech Palette)
    c_blue = "#1f77b4"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_purple = "#9467bd"
    c_red = "#d62728"
    c_gray = "#f4f4f4"
    c_box_edge = "#333333"

    # Title
    ax.text(7, 6.5, "End-to-End Palm-Vein Recognition Architecture", fontsize=16, fontweight="bold", ha="center")

    # Box 1: Input
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.2), 2.0, 2.0, boxstyle="round,pad=0.2", facecolor="#e8f4f8", edgecolor=c_blue, linewidth=2))
    ax.text(1.5, 4.5, "1. Input Stream\n(Raw Hand Image)", fontsize=10, fontweight="bold", ha="center", va="center")
    ax.text(1.5, 3.8, "RGB / NIR Sensor\nArbitrary size", fontsize=8.5, ha="center", va="center", color="#555555")

    # Arrow 1 -> 2
    ax.annotate("", xy=(3.0, 4.2), xytext=(2.5, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Box 2: Stage 1 SSD Detector
    ax.add_patch(patches.FancyBboxPatch((3.0, 2.6), 2.5, 3.2, boxstyle="round,pad=0.2", facecolor="#fff2e6", edgecolor=c_orange, linewidth=2))
    ax.text(4.25, 5.4, "Stage 1: ROI Detection", fontsize=10, fontweight="bold", ha="center", va="center", color=c_orange)
    ax.text(4.25, 4.6, "SSDLite320\n(MobileNetV3-Large)", fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax.text(4.25, 3.7, "- Input: 320x320\n- Anchor Box Head\n- Non-Max Suppression\n- BBox Expansion (10%)", fontsize=8, ha="center", va="center", color="#333333")
    ax.text(4.25, 2.9, "-> Output: Cropped Palm ROI", fontsize=8, fontweight="bold", ha="center", va="center", color="#222222")

    # Arrow 2 -> 3
    ax.annotate("", xy=(6.0, 4.2), xytext=(5.5, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Box 3: Stage 2 Enhancement
    ax.add_patch(patches.FancyBboxPatch((6.0, 2.6), 2.3, 3.2, boxstyle="round,pad=0.2", facecolor="#eafaf1", edgecolor=c_green, linewidth=2))
    ax.text(7.15, 5.4, "Stage 2: Enhancement", fontsize=10, fontweight="bold", ha="center", va="center", color=c_green)
    ax.text(7.15, 4.6, "VeinEnhancer", fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax.text(7.15, 3.7, "- Grayscale conversion\n- Resize to 224x224\n- CLAHE (clip=2.5)\n- Gaussian Filtering", fontsize=8, ha="center", va="center", color="#333333")
    ax.text(7.15, 2.9, "-> Output: Enhanced Vein Pattern", fontsize=8, fontweight="bold", ha="center", va="center", color="#222222")

    # Arrow 3 -> 4
    ax.annotate("", xy=(8.8, 4.2), xytext=(8.3, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Box 4: Stage 3 Deep Metric Learning
    ax.add_patch(patches.FancyBboxPatch((8.8, 2.6), 2.5, 3.2, boxstyle="round,pad=0.2", facecolor="#f3e8f8", edgecolor=c_purple, linewidth=2))
    ax.text(10.05, 5.4, "Stage 3: Deep Metric", fontsize=10, fontweight="bold", ha="center", va="center", color=c_purple)
    ax.text(10.05, 4.6, "PalmVeinNet", fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax.text(10.05, 3.7, "- EfficientNet-B0 Backbone\n- BN + Dropout + FC\n- L2 Feature Normalization\n- ArcFace Head (Margin=0.5)", fontsize=8, ha="center", va="center", color="#333333")
    ax.text(10.05, 2.9, "-> Output: 512-d Embedding", fontsize=8, fontweight="bold", ha="center", va="center", color="#222222")

    # Arrow 4 -> 5
    ax.annotate("", xy=(11.8, 4.2), xytext=(11.3, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # Box 5: Stage 4 Matching & Biometrics
    ax.add_patch(patches.FancyBboxPatch((11.8, 2.6), 2.0, 3.2, boxstyle="round,pad=0.2", facecolor="#fbeaea", edgecolor=c_red, linewidth=2))
    ax.text(12.8, 5.4, "Stage 4: Matching", fontsize=10, fontweight="bold", ha="center", va="center", color=c_red)
    ax.text(12.8, 4.6, "Biometric Matcher", fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax.text(12.8, 3.7, "- Cosine Similarity\n- 1:1 Verification\n- 1:N Identification\n- Metric: EER / FAR / FRR", fontsize=8, ha="center", va="center", color="#333333")
    ax.text(12.8, 2.9, "-> Result: Match / ID", fontsize=8, fontweight="bold", ha="center", va="center", color="#222222")

    # Pipeline Flow Details Bottom Banner
    ax.add_patch(patches.Rectangle((0.5, 0.8), 13.3, 1.3, facecolor="#fafafa", edgecolor="#cccccc", linewidth=1.5, linestyle="--"))
    ax.text(0.7, 1.7, "Pipeline Characteristics & Technical Specifications:", fontsize=9, fontweight="bold", color="#333333")
    ax.text(0.7, 1.3, "• Edge-Friendly Inference: SSDLite MobileNetV3 + EfficientNet-B0 optimized for real-time mobile/embedded edge devices.", fontsize=8, color="#555555")
    ax.text(0.7, 1.0, "• Robust Biometrics: CLAHE suppresses non-uniform skin illumination; ArcFace ensures hyperspherical angular margin separation.", fontsize=8, color="#555555")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Diagram saved to {out_path}")

if __name__ == "__main__":
    draw_pipeline_diagram()
