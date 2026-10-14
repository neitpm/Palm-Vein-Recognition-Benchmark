# -*- coding: utf-8 -*-
"""
High-legibility, modern card-based architecture diagram for dual pipelines.
Output: benchmark_comparison/dual_architecture_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_polished_dual_architectures(out_path="benchmark_comparison/dual_architecture_comparison.png"):
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Brand Colors
    c_ssd = "#059669"        # Deep emerald
    c_ssd_bg = "#ecfdf5"     # Soft light green
    c_rcnn = "#2563eb"       # Royal blue
    c_rcnn_bg = "#eff6ff"    # Soft light blue
    c_card_bg = "#ffffff"
    c_border = "#e2e8f0"
    c_text_dark = "#0f172a"
    c_text_muted = "#475569"

    # Header
    ax.text(8.0, 9.4, "Dual Detection Architectures & Biometric Palm ROI Pipeline", 
            fontsize=17, fontweight="bold", color=c_text_dark, ha="center")
    ax.text(8.0, 9.0, "Empirical comparison of One-Stage (Edge) vs. Two-Stage (Server) pipelines for downstream vein recognition", 
            fontsize=11, color=c_text_muted, ha="center")

    # ==================== PIPELINE 1: SSDLite-MobileNetV3 (One-Stage) ====================
    ax.add_patch(patches.FancyBboxPatch((0.5, 4.8), 15.0, 3.8, boxstyle="round,pad=0.25", 
                                        facecolor=c_ssd_bg, edgecolor=c_ssd, linewidth=2.0))
    ax.text(1.0, 8.2, "PIPELINE 1: One-Stage SSDLite-MobileNetV3 (Edge-Optimized Real-Time)", 
            fontsize=13, fontweight="bold", color=c_ssd)

    # Step 1.1: Input
    ax.add_patch(patches.FancyBboxPatch((1.0, 5.2), 2.5, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5))
    ax.text(2.25, 7.3, "1. Input Stream", fontsize=11, fontweight="bold", color=c_text_dark, ha="center")
    ax.text(2.25, 6.2, "• RGB / NIR Sensor\n• Fixed: 320 x 320 px\n• ImageNet Normalized\n• Memory: Low footprint", 
            fontsize=9.5, color=c_text_muted, ha="center", va="center")

    ax.annotate("", xy=(4.1, 6.5), xytext=(3.6, 6.5), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#64748b", mutation_scale=15))

    # Step 1.2: Backbone
    ax.add_patch(patches.FancyBboxPatch((4.1, 5.2), 3.2, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5))
    ax.text(5.7, 7.3, "2. MobileNetV3-Large", fontsize=11, fontweight="bold", color=c_text_dark, ha="center")
    ax.text(5.7, 6.2, "• Inverted Residual Blocks\n• Depthwise Separable Conv\n• Squeeze-and-Excitation (SE)\n• Hard-Swish Activation", 
            fontsize=9.5, color=c_text_muted, ha="center", va="center")

    ax.annotate("", xy=(7.9, 6.5), xytext=(7.4, 6.5), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#64748b", mutation_scale=15))

    # Step 1.3: SSDLite Head
    ax.add_patch(patches.FancyBboxPatch((7.9, 5.2), 3.4, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5))
    ax.text(9.6, 7.3, "3. SSDLite Predictor", fontsize=11, fontweight="bold", color=c_text_dark, ha="center")
    ax.text(9.6, 6.2, "• Multi-scale Anchor Heads\n• Single-Shot Box Regression\n• Direct Soft-NMS Post-filter\n• +10% Adaptive BBox Margin", 
            fontsize=9.5, color=c_text_muted, ha="center", va="center")

    ax.annotate("", xy=(11.9, 6.5), xytext=(11.4, 6.5), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#64748b", mutation_scale=15))

    # Step 1.4: Output Summary
    ax.add_patch(patches.FancyBboxPatch((11.9, 5.2), 3.3, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_ssd, linewidth=2.0))
    ax.text(13.55, 7.3, "4. Output & Edge Profile", fontsize=11, fontweight="bold", color=c_ssd, ha="center")
    ax.text(13.55, 6.2, "• Model Size: ~15 MB\n• Latency: ~20 ms (CPU)\n• F1-Score: 0.9994 (99.9%)\n• Cropped Palm ROI", 
            fontsize=9.5, fontweight="bold", color=c_text_dark, ha="center", va="center")

    # ==================== PIPELINE 2: Faster R-CNN ResNet-50 (Two-Stage) ====================
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 15.0, 3.8, boxstyle="round,pad=0.25", 
                                        facecolor=c_rcnn_bg, edgecolor=c_rcnn, linewidth=2.0))
    ax.text(1.0, 3.9, "PIPELINE 2: Two-Stage Faster R-CNN ResNet-50 (Server-Side High-Capacity)", 
            fontsize=13, fontweight="bold", color=c_rcnn)

    # Step 2.1: Input
    ax.add_patch(patches.FancyBboxPatch((1.0, 0.9), 2.5, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5))
    ax.text(2.25, 3.0, "1. Input Stream", fontsize=11, fontweight="bold", color=c_text_dark, ha="center")
    ax.text(2.25, 1.9, "• RGB / NIR Sensor\n• Scale: 600 - 1000 px\n• High-resolution inputs\n• High VRAM consumption", 
            fontsize=9.5, color=c_text_muted, ha="center", va="center")

    ax.annotate("", xy=(4.1, 2.2), xytext=(3.6, 2.2), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#64748b", mutation_scale=15))

    # Step 2.2: Backbone FPN
    ax.add_patch(patches.FancyBboxPatch((4.1, 0.9), 3.2, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5))
    ax.text(5.7, 3.0, "2. ResNet-50 + FPN", fontsize=11, fontweight="bold", color=c_text_dark, ha="center")
    ax.text(5.7, 1.9, "• 50 Residual Conv Layers\n• Feature Pyramid Network\n• Rich Semantic Encodings\n• Heavy computational cost", 
            fontsize=9.5, color=c_text_muted, ha="center", va="center")

    ax.annotate("", xy=(7.9, 2.2), xytext=(7.4, 2.2), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#64748b", mutation_scale=15))

    # Step 2.3: Stage 1 RPN + RoIAlign
    ax.add_patch(patches.FancyBboxPatch((7.9, 0.9), 3.4, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5))
    ax.text(9.6, 3.0, "3. RPN & RoIAlign Head", fontsize=11, fontweight="bold", color=c_text_dark, ha="center")
    ax.text(9.6, 1.9, "• Dense Proposal Generation\n• Bilinear RoIAlign Pooling\n• Two-stage BBox refinement\n• Redundant candidate boxes", 
            fontsize=9.5, color=c_text_muted, ha="center", va="center")

    ax.annotate("", xy=(11.9, 2.2), xytext=(11.4, 2.2), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#64748b", mutation_scale=15))

    # Step 2.4: Output Summary
    ax.add_patch(patches.FancyBboxPatch((11.9, 0.9), 3.3, 2.6, boxstyle="round,pad=0.15", 
                                        facecolor=c_card_bg, edgecolor=c_rcnn, linewidth=2.0))
    ax.text(13.55, 3.0, "4. Output & Server Profile", fontsize=11, fontweight="bold", color=c_rcnn, ha="center")
    ax.text(13.55, 1.9, "• Model Size: ~165 MB\n• Latency: ~120 ms (Heavy)\n• F1-Score: 0.4545 (Over-proposals)\n• mAP@0.3: 1.0000", 
            fontsize=9.5, fontweight="bold", color=c_text_dark, ha="center", va="center")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Re-generated polished dual architecture diagram at {out_path}")

if __name__ == "__main__":
    draw_polished_dual_architectures()
