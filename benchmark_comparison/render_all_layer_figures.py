# -*- coding: utf-8 -*-
"""
Render exact 3D/Isometric neural network layer blocks (PlotNeuralNet / LaTeX style).
Redesigned with clean proportions, zero text overlap, high contrast, and publication-ready layout.

Generates:
1. docs/figures/1_mobilenetv3_bneck_block.png
2. docs/figures/2_ssdlite_mobilenetv3_architecture.png
3. docs/figures/3_efficientnet_b0_backbone.png
4. docs/figures/4_fasterrcnn_resnet50_fpn.png
5. docs/figures/5_fasterrcnn_efficientnetb0_fpn.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np


def draw_3d_box(ax, x, y, dx, dy, dz, face_color="#dbeafe", edge_color="#2563eb", alpha=0.9):
    """Draw isometric 3D block with clean illumination facets."""
    # Front face
    front = patches.Polygon([
        [x, y], [x + dx, y], [x + dx, y + dy], [x, y + dy]
    ], closed=True, facecolor=face_color, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=3)
    ax.add_patch(front)

    # Top face
    top = patches.Polygon([
        [x, y + dy], [x + dx, y + dy], [x + dx + dz, y + dy + dz * 0.6], [x + dz, y + dy + dz * 0.6]
    ], closed=True, facecolor=face_color, edgecolor=edge_color, lw=1.2, alpha=alpha * 0.82, zorder=2)
    ax.add_patch(top)

    # Right face
    right = patches.Polygon([
        [x + dx, y], [x + dx + dz, y + dz * 0.6], [x + dx + dz, y + dy + dz * 0.6], [x + dx, y + dy]
    ], closed=True, facecolor=face_color, edgecolor=edge_color, lw=1.2, alpha=alpha * 0.70, zorder=2)
    ax.add_patch(right)


def render_fig1_mobilenetv3_block(out_path="docs/figures/1_mobilenetv3_bneck_block.png"):
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # Titles - placed safely at top with ample clearance
    ax.text(0.6, 6.4, "MobileNetV3 Inverted Residual Block", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(0.6, 6.0, "Expansion (1x1) → Depthwise Conv (3x3) → Squeeze-and-Excitation → Linear Bottleneck (1x1) + Residual", 
            fontsize=10, color="#475569")

    # Y baseline for main backbone blocks is 1.6
    # 1. Input Tensor
    draw_3d_box(ax, 0.8, 1.6, 0.35, 2.2, 0.55, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(0.97, 1.2, "Input\n$H \\times W \\times C$", fontsize=8.5, ha="center", color="#1e293b", fontweight="bold")

    # 1x1 Conv kernel indicator
    ax.add_patch(patches.Rectangle((0.95, 2.5), 0.2, 0.25, facecolor="#ef4444", edgecolor="#b91c1c", zorder=5))
    ax.text(0.97, 4.05, "1×1 Conv\n(Expand)", ha="center", fontsize=8.5, fontweight="bold", color="#0284c7")

    # Arrow to Expanded Dwise
    ax.annotate("", xy=(2.5, 2.7), xytext=(1.8, 2.7), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563eb"))

    # 2. Expanded Feature Map
    draw_3d_box(ax, 2.6, 1.4, 1.1, 2.7, 0.6, face_color="#93c5fd", edge_color="#1d4ed8")
    ax.text(3.15, 0.95, "Expanded Tensor\n$H \\times W \\times (e \\cdot C)$", fontsize=8.5, ha="center", color="#1e293b", fontweight="bold")
    # 3x3 Dwise kernel
    ax.add_patch(patches.Rectangle((3.1, 2.6), 0.45, 0.45, facecolor="#ef4444", edgecolor="#b91c1c", zorder=5))
    ax.text(3.32, 2.82, "3×3", fontsize=7.5, color="#ffffff", fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(3.15, 4.35, "3×3 Depthwise\n+ Hard-Swish", fontsize=8.5, fontweight="bold", ha="center", color="#0284c7")

    # Arrow to Dwise Output
    ax.annotate("", xy=(5.0, 2.7), xytext=(4.3, 2.7), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563eb"))

    # 3. Depthwise Output (feeds into SE branch below)
    draw_3d_box(ax, 5.1, 1.4, 1.1, 2.7, 0.6, face_color="#60a5fa", edge_color="#1e40af")
    ax.text(5.65, 4.35, "Depthwise\nFeature Map", fontsize=8.5, fontweight="bold", ha="center", color="#0284c7")

    # 4. Squeeze-and-Excitation (SE) Branch - distinctly positioned in lower sub-region
    # Arrow dropping down to SE
    ax.annotate("", xy=(5.65, 0.8), xytext=(5.65, 1.3), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))
    ax.text(5.5, 1.05, "Pool", fontsize=7.5, ha="right", color="#475569", fontweight="bold")

    # SE block containers
    ax.add_patch(patches.FancyBboxPatch((5.8, 0.35), 2.7, 0.65, boxstyle="round,pad=0.08", 
                                        facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.2))
    ax.text(7.15, 0.82, "Squeeze-and-Excitation (SE)", fontsize=8, fontweight="bold", color="#334155", ha="center")
    
    # FC1 -> ReLU -> FC2 -> Hard-Sigmoid
    ax.add_patch(patches.Rectangle((6.0, 0.42), 0.6, 0.3, facecolor="#e2e8f0", edgecolor="#64748b"))
    ax.text(6.3, 0.57, "FC $C/4$", fontsize=7, ha="center", va="center")
    ax.annotate("", xy=(6.9, 0.57), xytext=(6.65, 0.57), arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748b"))

    ax.add_patch(patches.Rectangle((6.95, 0.42), 0.6, 0.3, facecolor="#fed7aa", edgecolor="#ea580c"))
    ax.text(7.25, 0.57, "FC $e\\cdot C$", fontsize=7, ha="center", va="center", color="#9a3412", fontweight="bold")
    ax.annotate("", xy=(7.9, 0.57), xytext=(7.6, 0.57), arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748b"))

    # Scale multiplier circle
    ax.add_patch(patches.Circle((8.1, 0.57), 0.16, facecolor="#fef08a", edgecolor="#ca8a04", lw=1.2))
    ax.text(8.1, 0.57, "⊗", fontsize=9.5, ha="center", va="center", color="#854d0e", fontweight="bold")

    # Arrow routing back up from SE
    ax.annotate("", xy=(8.1, 1.3), xytext=(8.1, 0.85), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))

    # 5. Linear Bottleneck 1x1
    draw_3d_box(ax, 7.6, 1.4, 1.0, 2.7, 0.6, face_color="#93c5fd", edge_color="#1d4ed8")
    ax.add_patch(patches.Rectangle((8.0, 2.6), 0.25, 0.35, facecolor="#ef4444", edgecolor="#b91c1c", zorder=5))
    ax.text(8.1, 4.35, "1×1 Conv\n(Linear Proj.)", fontsize=8.5, fontweight="bold", ha="center", color="#0284c7")

    # Arrow to Output
    ax.annotate("", xy=(9.7, 2.7), xytext=(9.1, 2.7), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563eb"))

    # 6. Output Tensor
    draw_3d_box(ax, 9.8, 1.6, 0.35, 2.2, 0.55, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(9.97, 1.2, "Output\n$H \\times W \\times C'$", fontsize=8.5, ha="center", color="#1e293b", fontweight="bold")

    # 7. Residual Skip Connection: Elevated High Arc Over the Top (Completely Clear of all blocks and labels)
    ax.annotate("", xy=(9.97, 4.9), xytext=(0.97, 4.9),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#0f172a", connectionstyle="arc3,rad=-0.16"))
    # Residual addition node
    ax.add_patch(patches.Circle((9.97, 4.9), 0.22, facecolor="#22c55e", edgecolor="#15803d", lw=1.5, zorder=6))
    ax.text(9.97, 4.9, "+", fontsize=13, fontweight="bold", ha="center", va="center", color="#ffffff", zorder=7)
    ax.text(5.5, 5.25, "Identity Residual Connection (Stride = 1 & $C = C'$)", 
            fontsize=9.5, fontweight="bold", ha="center", color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#94a3b8", lw=0.8))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


def render_fig2_ssdlite_architecture(out_path="docs/figures/2_ssdlite_mobilenetv3_architecture.png"):
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(0.6, 5.7, "SSDLite320 with MobileNetV3-Large Backbone", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(0.6, 5.3, "Pyramid feature maps from depthwise separable extra layers feeding light detection heads", 
            fontsize=10, color="#475569")

    # Input image
    draw_3d_box(ax, 0.7, 1.8, 0.35, 2.6, 0.5, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(0.87, 1.3, "Input\n$320\\times 320\\times 3$", fontsize=8.5, fontweight="bold", ha="center", color="#1e293b")

    # MobileNetV3 Inverted Residual blocks (pyramid hierarchy)
    blocks = [
        (1.5, 2.0, 0.45, 2.2, "$160\\times 160$"),
        (2.3, 2.2, 0.50, 1.8, "$80\\times 80$"),
        (3.1, 2.4, 0.55, 1.4, "$40\\times 40$"),
        (4.0, 2.5, 0.60, 1.2, "$20\\times 20$ (C4)"),
        (5.0, 2.6, 0.65, 1.0, "$10\\times 10$ (C5)"),
    ]
    for x, y, dx, dy, lbl in blocks:
        draw_3d_box(ax, x, y, dx, dy, 0.45, face_color="#f1f5f9", edge_color="#64748b")
        ax.text(x + dx/2, y - 0.4, lbl, fontsize=7.5, ha="center", color="#334155")

    ax.text(3.5, 1.0, "MobileNetV3 Inverted Residual Backbone", fontsize=9.5, fontweight="bold", ha="center", color="#1e293b")

    # Extra SSDLite Layers (Lightweight Depthwise Separable Convs)
    extra_blocks = [
        (6.1, 2.7, 0.5, 0.8, "$5\\times 5$"),
        (7.0, 2.8, 0.5, 0.6, "$3\\times 3$"),
        (7.9, 2.85, 0.5, 0.5, "$2\\times 2$"),
        (8.8, 2.9, 0.5, 0.4, "$1\\times 1$"),
    ]
    for x, y, dx, dy, lbl in extra_blocks:
        draw_3d_box(ax, x, y, dx, dy, 0.4, face_color="#fed7aa", edge_color="#ea580c")
        ax.text(x + dx/2, y - 0.4, lbl, fontsize=7.5, ha="center", color="#9a3412", fontweight="bold")

    ax.text(7.6, 1.0, "SSDLite Extra Feature Pyramid (Depthwise Convs)", fontsize=9.5, fontweight="bold", ha="center", color="#c2410c")

    # Detection Heads bracketed at top
    ax.add_patch(patches.FancyBboxPatch((4.5, 4.3), 5.5, 0.85, boxstyle="round,pad=0.1", facecolor="#ecfdf5", edgecolor="#10b981", lw=1.5))
    ax.text(7.25, 4.75, "SSDLite Depthwise Separable Detection Heads", fontsize=10.5, fontweight="bold", ha="center", color="#065f46")
    ax.text(7.25, 4.45, "6 Pyramid Scales $\\times$ (Classification 3×3 Dwise + Regression 3×3 Dwise)", fontsize=8.5, ha="center", color="#047857")

    # Routing arrows from features up to detection head
    for fx in [4.3, 5.3, 6.35, 7.25, 8.15, 9.05]:
        ax.annotate("", xy=(fx, 4.3), xytext=(fx, 3.6), arrowprops=dict(arrowstyle="->", lw=1.2, color="#10b981"))

    # Final Output box on the right
    ax.add_patch(patches.FancyBboxPatch((10.3, 2.4), 1.4, 2.0, boxstyle="round,pad=0.1", facecolor="#f8fafc", edgecolor="#0f172a", lw=1.5))
    ax.text(11.0, 3.8, "Predicted\nPalm ROI", fontsize=9.5, fontweight="bold", ha="center", color="#0f172a")
    ax.text(11.0, 3.0, "Class: Palm (1)\nScores\nBoxes $(x,y,w,h)$", fontsize=8, ha="center", color="#475569")
    ax.annotate("", xy=(10.3, 3.4), xytext=(9.7, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color="#0f172a"))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


def render_fig3_efficientnet_backbone(out_path="docs/figures/3_efficientnet_b0_backbone.png"):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(0.6, 6.1, "EfficientNet-B0 Hierarchical Backbone (MBConv1 / MBConv6)", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(0.6, 5.7, "Compound scaled inverted bottlenecks with squeeze-and-excitation attention at each stage", 
            fontsize=10, color="#475569")

    # Input image
    draw_3d_box(ax, 0.6, 3.6, 0.35, 1.6, 0.45, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(0.77, 3.0, "Input\n$224\\times 224\\times 3$", fontsize=8, fontweight="bold", ha="center", color="#1e293b")

    # Stage 0 (Stem Conv3x3)
    draw_3d_box(ax, 1.4, 3.7, 0.35, 1.5, 0.4, face_color="#3b82f6", edge_color="#1d4ed8")
    ax.text(1.57, 3.1, "Stem Conv\n$112\\times 112\\times 32$", fontsize=7.5, ha="center", color="#1e293b")

    # Stage 1: MBConv1 (1 block)
    draw_3d_box(ax, 2.3, 3.8, 0.35, 1.3, 0.4, face_color="#64748b", edge_color="#334155")
    ax.text(2.47, 3.1, "MBConv1\n$112\\times 112\\times 16$", fontsize=7.5, ha="center", color="#1e293b")

    # Stage 2: MBConv6 (2 blocks)
    draw_3d_box(ax, 3.3, 3.9, 0.35, 1.1, 0.4, face_color="#fb923c", edge_color="#c2410c")
    draw_3d_box(ax, 3.8, 3.9, 0.35, 1.1, 0.4, face_color="#fb923c", edge_color="#c2410c")
    ax.text(3.8, 3.1, "Stage 2 (×2)\n$56\\times 56\\times 24$", fontsize=7.5, ha="center", color="#9a3412")

    # Stage 3: MBConv6 (2 blocks)
    draw_3d_box(ax, 4.8, 4.0, 0.35, 0.95, 0.4, face_color="#facc15", edge_color="#ca8a04")
    draw_3d_box(ax, 5.3, 4.0, 0.35, 0.95, 0.4, face_color="#facc15", edge_color="#ca8a04")
    ax.text(5.3, 3.1, "Stage 3 (×2)\n$28\\times 28\\times 40$", fontsize=7.5, ha="center", color="#854d0e")

    # Stage 4: MBConv6 (3 blocks)
    draw_3d_box(ax, 6.3, 4.0, 0.35, 0.95, 0.4, face_color="#60a5fa", edge_color="#2563eb")
    draw_3d_box(ax, 6.8, 4.0, 0.35, 0.95, 0.4, face_color="#60a5fa", edge_color="#2563eb")
    draw_3d_box(ax, 7.3, 4.0, 0.35, 0.95, 0.4, face_color="#60a5fa", edge_color="#2563eb")
    ax.text(7.0, 3.1, "Stage 4 (×3)\n$28\\times 28\\times 80$", fontsize=7.5, ha="center", color="#1e40af")

    # Route down to lower row
    ax.annotate("", xy=(8.7, 2.4), xytext=(8.7, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color="#2563eb"))

    # Stage 5: MBConv6 (3 blocks)
    draw_3d_box(ax, 8.2, 1.3, 0.35, 0.9, 0.4, face_color="#94a3b8", edge_color="#475569")
    draw_3d_box(ax, 7.6, 1.3, 0.35, 0.9, 0.4, face_color="#94a3b8", edge_color="#475569")
    draw_3d_box(ax, 7.0, 1.3, 0.35, 0.9, 0.4, face_color="#94a3b8", edge_color="#475569")
    ax.text(7.8, 0.7, "Stage 5 (×3)\n$14\\times 14\\times 112$", fontsize=7.5, ha="center", color="#334155")

    # Stage 6: MBConv6 (3 blocks)
    draw_3d_box(ax, 6.0, 1.4, 0.35, 0.7, 0.4, face_color="#4ade80", edge_color="#16a34a")
    draw_3d_box(ax, 5.4, 1.4, 0.35, 0.7, 0.4, face_color="#4ade80", edge_color="#16a34a")
    draw_3d_box(ax, 4.8, 1.4, 0.35, 0.7, 0.4, face_color="#4ade80", edge_color="#16a34a")
    ax.text(5.6, 0.7, "Stage 6 (×3)\n$7\\times 7\\times 192$", fontsize=7.5, ha="center", color="#15803d")

    # Stage 7: MBConv6 (1 block)
    draw_3d_box(ax, 3.8, 1.45, 0.4, 0.65, 0.4, face_color="#cbd5e1", edge_color="#475569")
    ax.text(4.0, 0.7, "Stage 7 (×1)\n$7\\times 7\\times 320$", fontsize=7.5, ha="center", color="#334155")

    # Head 1x1 Conv + Pooling
    draw_3d_box(ax, 2.6, 1.5, 0.4, 0.55, 0.4, face_color="#f87171", edge_color="#b91c1c")
    ax.text(2.8, 0.7, "Conv 1×1\n$7\\times 7\\times 1280$", fontsize=7.5, ha="center", color="#991b1b")

    # Final Global Pooling & Embedding
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 1.2, 1.4, boxstyle="round,pad=0.1", facecolor="#f8fafc", edgecolor="#0f172a", lw=1.5))
    ax.text(1.4, 2.0, "Global AvgPool", fontsize=8.5, fontweight="bold", ha="center", color="#0f172a")
    ax.text(1.4, 1.4, "1280-d Vector\nFeature Output", fontsize=8, ha="center", color="#475569")
    ax.annotate("", xy=(2.0, 1.8), xytext=(2.6, 1.8), arrowprops=dict(arrowstyle="->", lw=1.8, color="#0f172a"))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")



def render_fig4_fasterrcnn_resnet50(out_path="docs/figures/4_fasterrcnn_resnet50_fpn.png"):
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(0.6, 5.7, "Faster R-CNN with ResNet-50-FPN Two-Stage Architecture", fontsize=15, fontweight="bold", color="#581c87")
    ax.text(0.6, 5.3, "Stage 1: Region Proposal Network (RPN) | Stage 2: RoIAlign + Fast R-CNN Classification & Regression Heads", 
            fontsize=10, color="#475569")

    # 1. Input Image
    draw_3d_box(ax, 0.6, 2.3, 0.35, 2.4, 0.5, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(0.77, 1.7, "Input Image\nMulti-Scale", fontsize=8.5, fontweight="bold", ha="center", color="#1e293b")
    ax.annotate("", xy=(1.5, 3.5), xytext=(1.1, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#581c87"))

    # 2. ResNet-50 Backbone Stages (C2, C3, C4, C5)
    draw_3d_box(ax, 1.6, 2.3, 0.6, 2.4, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 2.4, 2.3, 0.6, 2.4, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 3.2, 2.3, 0.6, 2.4, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(2.7, 1.7, "ResNet-50 Backbone\n(C2, C3, C4, C5 Stages)", fontsize=9.5, fontweight="bold", ha="center", color="#334155")

    # 3. Feature Pyramid Network (FPN) Block
    ax.annotate("", xy=(4.4, 3.5), xytext=(4.0, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#581c87"))
    draw_3d_box(ax, 4.5, 2.1, 0.8, 2.8, 0.5, face_color="#e0e7ff", edge_color="#4338ca")
    ax.text(4.9, 4.5, "Feature\nPyramid\n(FPN)", fontsize=10, fontweight="bold", ha="center", color="#312e81")

    # Separate, non-overlapping FPN Levels display placed safely below
    fpn_levels = [("P2", "#bfdbfe"), ("P3", "#86efac"), ("P4", "#facc15"), ("P5", "#f87171")]
    for idx, (lbl, col) in enumerate(fpn_levels):
        ax.add_patch(patches.Rectangle((4.2 + idx * 0.45, 1.2), 0.35, 0.3, facecolor=col, edgecolor="#334155", lw=1.0))
        ax.text(4.37 + idx * 0.45, 1.35, lbl, fontsize=8, ha="center", va="center", fontweight="bold")
    ax.text(5.0, 0.75, "256 channels per level (P2..P5)", fontsize=8, ha="center", color="#475569")

    # 4. Region Proposal Network (RPN)
    ax.annotate("", xy=(6.0, 3.5), xytext=(5.5, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#581c87"))
    ax.add_patch(patches.FancyBboxPatch((6.1, 2.5), 1.4, 2.0, boxstyle="round,pad=0.1", facecolor="#f8fafc", edgecolor="#0284c7", lw=1.8))
    ax.text(6.8, 4.1, "RPN", fontsize=12, fontweight="bold", ha="center", color="#0369a1")
    ax.text(6.8, 3.4, "Anchor Generation\nObjectness Logits\nProposal BBoxes", fontsize=8, ha="center", color="#334155")
    ax.text(6.8, 2.7, "Top 1000 Propos.", fontsize=8, fontweight="bold", ha="center", color="#0284c7")

    # 5. RoIAlign Block
    ax.annotate("", xy=(8.1, 3.5), xytext=(7.6, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#581c87"))
    draw_3d_box(ax, 8.2, 2.4, 0.9, 2.2, 0.5, face_color="#fef3c7", edge_color="#d97706")
    ax.text(8.65, 3.6, "RoIAlign\n$7\\times 7$", fontsize=10.5, fontweight="bold", ha="center", color="#92400e")
    ax.text(8.65, 1.7, "Bilinear Sampling\n(No Quantization)", fontsize=8, ha="center", color="#b45309")

    # 6. Two-Stage Fast R-CNN Head & Final Output
    ax.annotate("", xy=(9.7, 3.5), xytext=(9.3, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#581c87"))
    ax.add_patch(patches.FancyBboxPatch((9.8, 2.2), 1.8, 2.6, boxstyle="round,pad=0.1", facecolor="#faf5ff", edgecolor="#9333ea", lw=1.8))
    ax.text(10.7, 4.4, "Fast R-CNN Head", fontsize=11, fontweight="bold", ha="center", color="#6b21a8")
    ax.text(10.7, 3.7, "Two 1024-d FC Layers\nClassification Logits\nBox Delta Regr.", fontsize=8, ha="center", color="#475569")
    
    # Final Result pill
    ax.add_patch(patches.Rectangle((10.0, 2.5), 1.4, 0.7, facecolor="#22c55e", edgecolor="#15803d", lw=1.2))
    ax.text(10.7, 2.85, "Palm ROI (Class 1)\nBoxes + Conf Scores", fontsize=7.5, fontweight="bold", ha="center", va="center", color="#ffffff")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


def render_fig5_fasterrcnn_efficientnetb0(out_path="docs/figures/5_fasterrcnn_efficientnetb0_fpn.png"):
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(0.6, 5.7, "Faster R-CNN with EfficientNet-B0-FPN Two-Stage Architecture", fontsize=15, fontweight="bold", color="#1d4ed8")
    ax.text(0.6, 5.3, "Stage 1: Region Proposal Network (RPN) | Stage 2: RoIAlign + Fast R-CNN Classification & Regression Heads", 
            fontsize=10, color="#475569")

    # 1. Input Image
    draw_3d_box(ax, 0.6, 2.3, 0.35, 2.4, 0.5, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(0.77, 1.7, "Input Image\nMulti-Scale", fontsize=8.5, fontweight="bold", ha="center", color="#1e293b")
    ax.annotate("", xy=(1.5, 3.5), xytext=(1.1, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1d4ed8"))

    # 2. EfficientNet-B0 Backbone (Stages 2, 3, 5, 6 extracted for FPN)
    draw_3d_box(ax, 1.6, 2.3, 0.6, 2.4, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 2.4, 2.3, 0.6, 2.4, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 3.2, 2.3, 0.6, 2.4, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(2.7, 1.7, "EfficientNet-B0 Backbone\n(C2, C3, C4, C5 Pyramids)", fontsize=9.5, fontweight="bold", ha="center", color="#334155")

    # 3. Feature Pyramid Network (FPN) Block
    ax.annotate("", xy=(4.4, 3.5), xytext=(4.0, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1d4ed8"))
    draw_3d_box(ax, 4.5, 2.1, 0.8, 2.8, 0.5, face_color="#dbeafe", edge_color="#2563eb")
    ax.text(4.9, 4.5, "Feature\nPyramid\n(FPN)", fontsize=10, fontweight="bold", ha="center", color="#1e40af")

    # FPN Levels list below
    fpn_levels = [("P2", "#bfdbfe"), ("P3", "#86efac"), ("P4", "#facc15"), ("P5", "#f87171")]
    for idx, (lbl, col) in enumerate(fpn_levels):
        ax.add_patch(patches.Rectangle((4.2 + idx * 0.45, 1.2), 0.35, 0.3, facecolor=col, edgecolor="#334155", lw=1.0))
        ax.text(4.37 + idx * 0.45, 1.35, lbl, fontsize=8, ha="center", va="center", fontweight="bold")
    ax.text(5.0, 0.75, "256 channels per level (P2..P5)", fontsize=8, ha="center", color="#475569")

    # 4. Region Proposal Network (RPN)
    ax.annotate("", xy=(6.0, 3.5), xytext=(5.5, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1d4ed8"))
    ax.add_patch(patches.FancyBboxPatch((6.1, 2.5), 1.4, 2.0, boxstyle="round,pad=0.1", facecolor="#f8fafc", edgecolor="#0284c7", lw=1.8))
    ax.text(6.8, 4.1, "RPN", fontsize=12, fontweight="bold", ha="center", color="#0369a1")
    ax.text(6.8, 3.4, "Anchor Generation\nObjectness Logits\nProposal BBoxes", fontsize=8, ha="center", color="#334155")
    ax.text(6.8, 2.7, "Top 1000 Propos.", fontsize=8, fontweight="bold", ha="center", color="#0284c7")

    # 5. RoIAlign Block
    ax.annotate("", xy=(8.1, 3.5), xytext=(7.6, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1d4ed8"))
    draw_3d_box(ax, 8.2, 2.4, 0.9, 2.2, 0.5, face_color="#fef3c7", edge_color="#d97706")
    ax.text(8.65, 3.6, "RoIAlign\n$7\\times 7$", fontsize=10.5, fontweight="bold", ha="center", color="#92400e")
    ax.text(8.65, 1.7, "Bilinear Sampling\n(No Quantization)", fontsize=8, ha="center", color="#b45309")

    # 6. Two-Stage Fast R-CNN Head & Final Output
    ax.annotate("", xy=(9.7, 3.5), xytext=(9.3, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1d4ed8"))
    ax.add_patch(patches.FancyBboxPatch((9.8, 2.2), 1.8, 2.6, boxstyle="round,pad=0.1", facecolor="#eff6ff", edgecolor="#2563eb", lw=1.8))
    ax.text(10.7, 4.4, "Fast R-CNN Head", fontsize=11, fontweight="bold", ha="center", color="#1d4ed8")
    ax.text(10.7, 3.7, "Two 1024-d FC Layers\nClassification Logits\nBox Delta Regr.", fontsize=8, ha="center", color="#475569")
    
    # Final Result pill
    ax.add_patch(patches.Rectangle((10.0, 2.5), 1.4, 0.7, facecolor="#22c55e", edgecolor="#15803d", lw=1.2))
    ax.text(10.7, 2.85, "Palm ROI (Class 1)\nBoxes + Conf Scores", fontsize=7.5, fontweight="bold", ha="center", va="center", color="#ffffff")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


if __name__ == "__main__":
    render_fig1_mobilenetv3_block()
    render_fig2_ssdlite_architecture()
    render_fig3_efficientnet_backbone()
    render_fig4_fasterrcnn_resnet50()
    render_fig5_fasterrcnn_efficientnetb0()
