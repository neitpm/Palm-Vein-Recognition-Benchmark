# -*- coding: utf-8 -*-
"""
Render exact 3D/Isometric neural network layer blocks (PlotNeuralNet / TeX style).
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

def draw_3d_box(ax, x, y, dx, dy, dz, face_color="#dbeafe", edge_color="#3b82f6", alpha=0.9):
    """Vẽ khối hộp 3D Isometric chuẩn phong cách bài báo Deep Learning."""
    # Front face
    front = patches.Polygon([
        [x, y], [x + dx, y], [x + dx, y + dy], [x, y + dy]
    ], closed=True, facecolor=face_color, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=3)
    ax.add_patch(front)

    # Top face
    top = patches.Polygon([
        [x, y + dy], [x + dx, y + dy], [x + dx + dz, y + dy + dz * 0.7], [x + dz, y + dy + dz * 0.7]
    ], closed=True, facecolor=face_color, edgecolor=edge_color, lw=1.2, alpha=alpha * 0.85, zorder=2)
    ax.add_patch(top)

    # Right face
    right = patches.Polygon([
        [x + dx, y], [x + dx + dz, y + dz * 0.7], [x + dx + dz, y + dy + dz * 0.7], [x + dx, y + dy]
    ], closed=True, facecolor=face_color, edgecolor=edge_color, lw=1.2, alpha=alpha * 0.75, zorder=2)
    ax.add_patch(right)


def render_fig1_mobilenetv3_block(out_path="docs/figures/1_mobilenetv3_bneck_block.png"):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(0.5, 4.4, "MobileNetV3 Inverted Residual Block", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(5.0, 3.8, "MobileNet V3 Block with SE Attention", fontsize=11, fontweight="bold", ha="center", color="#334155")

    # Layer 1: Input feature (thin block)
    draw_3d_box(ax, 0.8, 1.2, 0.3, 2.2, 0.6, face_color="#93c5fd", edge_color="#2563eb")
    # Kernel 1x1
    ax.add_patch(patches.Rectangle((0.95, 2.1), 0.25, 0.3, facecolor="#ef4444", alpha=0.8, zorder=5))
    ax.text(1.3, 1.9, "1x1, NL", fontsize=8, color="#1e293b", fontweight="bold")

    # Projection ray
    ax.add_patch(patches.Polygon([[1.2, 2.4], [2.2, 3.2], [2.2, 1.4]], facecolor="#f87171", alpha=0.3, zorder=2))

    # Layer 2: Expanded Dwise feature
    draw_3d_box(ax, 2.2, 1.0, 1.2, 2.6, 0.6, face_color="#60a5fa", edge_color="#1d4ed8")
    # Kernel 3x3
    ax.add_patch(patches.Rectangle((2.9, 2.1), 0.4, 0.4, facecolor="#ef4444", alpha=0.8, zorder=5))
    ax.text(3.1, 2.25, "3x3", fontsize=7.5, color="#ffffff", fontweight="bold", ha="center")
    ax.text(3.7, 1.9, "Dwise, NL", fontsize=8, color="#1e293b", fontweight="bold")

    # Layer 3: Depthwise output
    draw_3d_box(ax, 4.3, 1.0, 1.2, 2.6, 0.6, face_color="#60a5fa", edge_color="#1d4ed8")

    # Squeeze-and-Excitation branch below
    ax.annotate("", xy=(5.0, 0.5), xytext=(5.0, 1.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))
    ax.text(5.2, 0.7, "Pool", fontsize=8, color="#475569")
    ax.add_patch(patches.Rectangle((4.8, 0.35), 0.8, 0.12, facecolor="#64748b"))
    ax.annotate("", xy=(6.0, 0.4), xytext=(5.6, 0.4), arrowprops=dict(arrowstyle="->", lw=1.2, color="#475569"))
    ax.text(5.8, 0.1, "FC, ReLU", fontsize=7.5, ha="center")
    ax.add_patch(patches.Rectangle((6.0, 0.35), 0.3, 0.12, facecolor="#64748b"))
    ax.annotate("", xy=(6.7, 0.4), xytext=(6.3, 0.4), arrowprops=dict(arrowstyle="->", lw=1.2, color="#475569"))
    ax.text(6.8, 0.1, "FC, hard-σ", fontsize=7.5, ha="center")
    ax.add_patch(patches.Rectangle((6.8, 0.35), 0.8, 0.12, facecolor="#64748b"))
    ax.annotate("", xy=(7.2, 1.0), xytext=(7.2, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))
    ax.text(7.4, 0.7, "⊗", fontsize=10, fontweight="bold")

    # Layer 4: Linear bottleneck 1x1
    draw_3d_box(ax, 6.7, 1.0, 1.2, 2.6, 0.6, face_color="#60a5fa", edge_color="#1d4ed8")
    ax.add_patch(patches.Rectangle((7.4, 2.1), 0.3, 0.3, facecolor="#ef4444", alpha=0.8, zorder=5))
    ax.text(8.0, 2.2, "1x1", fontsize=8, color="#1e293b", fontweight="bold")

    # Layer 5: Output with residual connection
    draw_3d_box(ax, 8.6, 1.2, 0.3, 2.2, 0.6, face_color="#e2e8f0", edge_color="#64748b", alpha=0.5)

    # Residual skip connection curve
    ax.annotate("", xy=(8.7, 1.0), xytext=(0.95, 1.0),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b", connectionstyle="arc3,rad=-0.25"))
    ax.text(4.8, 0.0, "+", fontsize=14, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


def render_fig2_ssdlite_architecture(out_path="docs/figures/2_ssdlite_mobilenetv3_architecture.png"):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(0.5, 4.4, "SSDLite with MobileNetV3 Backbone", fontsize=14, fontweight="bold", color="#0f172a")

    # Input Image Box
    draw_3d_box(ax, 0.5, 1.5, 0.25, 2.6, 0.6, face_color="#93c5fd", edge_color="#1d4ed8")
    ax.text(0.62, 2.8, "MobileNetV3", fontsize=9, fontweight="bold", rotation=90, va="center", ha="center")

    # Inverted Residual Blocks (decreasing size, increasing depth)
    draw_3d_box(ax, 1.3, 1.7, 0.4, 2.2, 0.5, face_color="#f1f5f9", edge_color="#64748b")
    draw_3d_box(ax, 2.0, 1.9, 0.45, 1.8, 0.5, face_color="#f1f5f9", edge_color="#64748b")
    draw_3d_box(ax, 2.8, 2.0, 0.5, 1.6, 0.5, face_color="#f1f5f9", edge_color="#64748b")
    draw_3d_box(ax, 3.6, 2.1, 0.55, 1.4, 0.5, face_color="#f1f5f9", edge_color="#64748b")

    ax.text(2.6, 1.2, "16 × 14 × 24\n56 × 112 × 16", fontsize=8.5, ha="center", color="#334155")
    ax.text(2.6, 0.6, "Inverted residual (bottleneck)", fontsize=9, fontweight="bold", ha="center")

    # Detection Head bracket
    ax.text(5.2, 3.8, "Det. Head", fontsize=9, fontweight="bold", ha="center",
            bbox=dict(boxstyle="square,pad=0.3", fc="#ffffff", ec="#1e293b"))

    # Conv blocks in Det. Head
    draw_3d_box(ax, 4.7, 2.3, 0.45, 0.9, 0.4, face_color="#f1f5f9", edge_color="#64748b")
    draw_3d_box(ax, 5.5, 2.3, 0.5, 0.9, 0.4, face_color="#86efac", edge_color="#16a34a") # Green conv
    ax.text(5.75, 2.7, "Conv", fontsize=8.5, fontweight="bold", ha="center")

    draw_3d_box(ax, 6.3, 2.3, 0.45, 0.9, 0.4, face_color="#f1f5f9", edge_color="#64748b")
    ax.text(6.52, 2.7, "Clas.", fontsize=8, ha="center")

    draw_3d_box(ax, 7.1, 2.3, 0.5, 0.9, 0.4, face_color="#f1f5f9", edge_color="#64748b")
    ax.text(7.35, 2.7, "Boxes", fontsize=8, ha="center")

    # Final feature map
    draw_3d_box(ax, 8.1, 2.0, 0.15, 1.4, 0.4, face_color="#e2e8f0", edge_color="#475569")
    ax.annotate("", xy=(8.0, 2.7), xytext=(7.6, 2.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1e293b"))

    # Adaptive pooling box below
    ax.annotate("", xy=(7.35, 1.5), xytext=(7.35, 2.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1e293b"))
    ax.text(7.6, 1.7, "Adaptive pooling", fontsize=8.5, fontweight="bold")
    draw_3d_box(ax, 7.0, 0.4, 0.7, 0.9, 0.3, face_color="#f8fafc", edge_color="#64748b")
    ax.text(7.35, 0.8, "Conv\n1x", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


def render_fig3_efficientnet_backbone(out_path="docs/figures/3_efficientnet_b0_backbone.png"):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(0.5, 4.6, "EfficientNet-B0 Layer Architecture & Feature Flow", fontsize=13, fontweight="bold", color="#0f172a")

    # Input image arrow
    ax.annotate("", xy=(0.9, 3.6), xytext=(0.2, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563eb"))
    ax.text(0.5, 3.8, "Input image\n224 × 224 × 3", fontsize=8, ha="center", color="#1e293b")

    # Row 1 (top): Left to Right
    # Stem: 112x112x32
    draw_3d_box(ax, 0.9, 2.8, 0.3, 1.7, 0.4, face_color="#3b82f6", edge_color="#1d4ed8")
    ax.text(1.3, 3.7, "112 × 112 × 32", fontsize=7.5, rotation=90, color="#1e293b")

    # Stage 1: 112x112x16
    draw_3d_box(ax, 1.8, 2.9, 0.25, 1.5, 0.4, face_color="#475569", edge_color="#1e293b")
    ax.text(2.2, 3.7, "112 × 112 × 16", fontsize=7.5, rotation=90, color="#1e293b")

    # Stage 2: 56x56x24 (2 blocks)
    draw_3d_box(ax, 2.7, 3.0, 0.3, 1.3, 0.4, face_color="#fb923c", edge_color="#c2410c")
    draw_3d_box(ax, 3.4, 3.0, 0.3, 1.3, 0.4, face_color="#fb923c", edge_color="#c2410c")
    ax.text(3.9, 3.7, "56 × 56 × 24", fontsize=7.5, rotation=90, color="#1e293b")

    # Stage 3: 28x28x40 (2 blocks)
    draw_3d_box(ax, 4.3, 3.1, 0.35, 1.1, 0.4, face_color="#facc15", edge_color="#ca8a04")
    draw_3d_box(ax, 5.0, 3.1, 0.35, 1.1, 0.4, face_color="#facc15", edge_color="#ca8a04")
    ax.text(5.5, 3.7, "28 × 28 × 40", fontsize=7.5, rotation=90, color="#1e293b")

    # Stage 4: 28x28x80 (3 blocks)
    draw_3d_box(ax, 5.9, 3.1, 0.35, 1.1, 0.4, face_color="#60a5fa", edge_color="#2563eb")
    draw_3d_box(ax, 6.6, 3.1, 0.35, 1.1, 0.4, face_color="#60a5fa", edge_color="#2563eb")
    draw_3d_box(ax, 7.3, 3.1, 0.35, 1.1, 0.4, face_color="#60a5fa", edge_color="#2563eb")
    ax.text(7.8, 3.7, "28 × 28 × 80", fontsize=7.5, rotation=90, color="#1e293b")

    # Turn down arrow
    draw_3d_box(ax, 8.2, 3.2, 0.4, 0.9, 0.4, face_color="#64748b", edge_color="#334155")
    ax.annotate("", xy=(8.4, 2.3), xytext=(8.4, 3.1), arrowprops=dict(arrowstyle="->", lw=2, color="#2563eb"))
    ax.text(8.7, 2.7, "14 × 14 × 112", fontsize=7.5, rotation=90, color="#1e293b")

    # Row 2 (bottom): Right to Left
    # Stage 5: 14x14x112 (3 blocks)
    draw_3d_box(ax, 7.9, 1.1, 0.4, 0.9, 0.4, face_color="#64748b", edge_color="#334155")
    draw_3d_box(ax, 7.2, 1.1, 0.4, 0.9, 0.4, face_color="#64748b", edge_color="#334155")
    draw_3d_box(ax, 6.5, 1.1, 0.4, 0.9, 0.4, face_color="#64748b", edge_color="#334155")
    ax.text(6.3, 1.6, "14 × 14 × 112", fontsize=7.5, rotation=90, color="#1e293b")

    # Stage 6: 7x7x192 (3 blocks)
    draw_3d_box(ax, 5.6, 1.2, 0.45, 0.7, 0.4, face_color="#4ade80", edge_color="#16a34a")
    draw_3d_box(ax, 4.9, 1.2, 0.45, 0.7, 0.4, face_color="#4ade80", edge_color="#16a34a")
    draw_3d_box(ax, 4.2, 1.2, 0.45, 0.7, 0.4, face_color="#4ade80", edge_color="#16a34a")
    ax.text(4.0, 1.6, "7 × 7 × 192", fontsize=7.5, rotation=90, color="#1e293b")

    # Stage 7: 7x7x320 (Conv Head)
    draw_3d_box(ax, 3.2, 1.2, 0.5, 0.7, 0.4, face_color="#cbd5e1", edge_color="#475569")
    ax.text(3.0, 1.6, "7 × 7 × 320", fontsize=7.5, rotation=90, color="#1e293b")

    # Pooling & Dropout: 1x1x1280
    ax.add_patch(patches.Rectangle((2.3, 0.7), 0.4, 1.8, facecolor="#ffffff", edgecolor="#f97316", lw=1.5))
    for cy in np.linspace(0.9, 2.3, 6):
        ax.add_patch(patches.Circle((2.5, cy), 0.12, facecolor="#ffffff", edgecolor="#f97316", lw=1.2))
    ax.text(2.1, 1.6, "1 × 1 × 1280", fontsize=7.5, rotation=90, color="#1e293b")

    # Output Vector
    ax.add_patch(patches.Rectangle((1.4, 0.8), 0.4, 1.5, facecolor="#ffffff", edgecolor="#0f172a", lw=1.5))
    for cy in np.linspace(1.0, 2.1, 4):
        ax.add_patch(patches.Circle((1.6, cy), 0.1, facecolor="#ffffff", edgecolor="#0f172a", lw=1.2))
    ax.text(1.6, 0.4, "Output", fontsize=8.5, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


def render_fig4_fasterrcnn_resnet50(out_path="docs/figures/4_fasterrcnn_resnet50_fpn.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Purple border frame
    ax.add_patch(patches.Rectangle((0.4, 0.3), 9.2, 4.8, fill=False, edgecolor="#9333ea", lw=2.5))
    ax.text(0.8, 4.8, "Faster R-CNN ResNet-50-FPN Architecture", fontsize=13, fontweight="bold", color="#581c87")

    # Input Image
    draw_3d_box(ax, 0.8, 2.5, 0.3, 2.0, 0.5, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(0.95, 3.5, "Input", fontsize=10, fontweight="bold", rotation=90, va="center", ha="center")
    ax.annotate("", xy=(1.9, 3.5), xytext=(1.4, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # ResNet-50 Backbone 3 stages
    draw_3d_box(ax, 1.9, 2.6, 0.6, 1.8, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 2.8, 2.6, 0.6, 1.8, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 3.7, 2.6, 0.6, 1.8, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(3.1, 2.2, "16 × 24 × 2048", fontsize=9, fontweight="bold", ha="center")
    ax.text(3.1, 1.6, "ResNet-50\nbackbone", fontsize=11, fontweight="bold", ha="center")

    # Stack of feature pyramids bottom left
    draw_3d_box(ax, 2.0, 1.1, 0.8, 0.3, 0.3, face_color="#f87171", edge_color="#991b1b") # Red P5
    draw_3d_box(ax, 2.0, 1.4, 0.8, 0.3, 0.3, face_color="#facc15", edge_color="#854d0e") # Yellow P4
    draw_3d_box(ax, 2.0, 1.7, 0.8, 0.3, 0.3, face_color="#86efac", edge_color="#166534") # Green P3
    draw_3d_box(ax, 2.0, 2.0, 0.8, 0.3, 0.3, face_color="#bfdbfe", edge_color="#1e40af") # Blue P2
    ax.text(3.2, 1.4, "56 × 56 × 256\n28 × 28 × 512\n14 × 14 × 1024\n7 × 7 × 2048", fontsize=7.5, color="#1e293b")

    # FPN Block
    draw_3d_box(ax, 4.9, 2.8, 0.6, 1.4, 0.5, face_color="#bfdbfe", edge_color="#2563eb")
    ax.text(5.2, 3.5, "FPN", fontsize=10, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(4.9, 3.5), xytext=(4.5, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # FPN levels list
    for idx, (lbl, col) in enumerate([("P2", "#bfdbfe"), ("P3", "#86efac"), ("P4", "#facc15"), ("P5", "#f87171")]):
        ax.add_patch(patches.Rectangle((4.9, 2.3 - idx * 0.3), 0.3, 0.25, facecolor=col, edgecolor="#334155"))
        ax.text(5.05, 2.42 - idx * 0.3, lbl, fontsize=7, ha="center", va="center", fontweight="bold")
    ax.text(6.0, 2.0, "Pyramid\nFeatures", fontsize=8.5, color="#334155")

    # RPN Block
    draw_3d_box(ax, 6.0, 2.7, 0.9, 1.6, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(6.45, 3.5, "RPN", fontsize=11, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(6.0, 3.5), xytext=(5.6, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # RoI Align Block
    draw_3d_box(ax, 7.3, 2.7, 0.9, 1.6, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(7.75, 3.6, "RoI\nAlign", fontsize=10.5, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(7.3, 3.5), xytext=(7.0, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # Output head arrow down
    ax.annotate("", xy=(7.75, 1.8), xytext=(7.75, 2.6), arrowprops=dict(arrowstyle="->", lw=2, color="#1e293b"))
    ax.add_patch(patches.Rectangle((7.2, 0.6), 1.3, 1.1, facecolor="#ffffff", edgecolor="#0f172a", lw=1.5))
    ax.text(7.85, 0.3, "Boxes, Labels, Scores", fontsize=9, fontweight="bold", ha="center")

    # Final Output Block right
    draw_3d_box(ax, 8.8, 2.7, 0.25, 1.6, 0.4, face_color="#e2e8f0", edge_color="#475569")
    ax.annotate("", xy=(8.8, 3.5), xytext=(8.4, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))
    ax.annotate("", xy=(8.9, 2.2), xytext=(8.9, 2.6), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))
    ax.text(8.9, 1.9, "Output", fontsize=10, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


def render_fig5_fasterrcnn_efficientnetb0(out_path="docs/figures/5_fasterrcnn_efficientnetb0_fpn.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300, facecolor="#ffffff")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Blue banner header
    ax.add_patch(patches.FancyBboxPatch((2.2, 4.8), 5.6, 0.6, boxstyle="round,pad=0.1", facecolor="#2563eb", edgecolor="none"))
    ax.text(5.0, 5.1, "Faster R-CNN EfficientNet-B0 Backbone", fontsize=11, fontweight="bold", color="#ffffff", ha="center")

    # Input Image
    draw_3d_box(ax, 0.8, 2.4, 0.3, 2.0, 0.5, face_color="#bfdbfe", edge_color="#1d4ed8")
    ax.text(0.95, 3.4, "Input", fontsize=10, fontweight="bold", rotation=90, va="center", ha="center")
    ax.annotate("", xy=(1.9, 3.4), xytext=(1.4, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # EfficientNet Backbone 3 stages
    draw_3d_box(ax, 1.9, 2.5, 0.6, 1.8, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 2.8, 2.5, 0.6, 1.8, 0.5, face_color="#f1f5f9", edge_color="#475569")
    draw_3d_box(ax, 3.7, 2.5, 0.6, 1.8, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(3.1, 2.1, "16 × 24 × 1280", fontsize=9, fontweight="bold", ha="center")
    ax.text(3.1, 1.5, "EfficientNet\nbackbone", fontsize=11, fontweight="bold", ha="center")

    # Feature pyramid stack bottom left
    draw_3d_box(ax, 2.0, 0.9, 0.8, 0.3, 0.3, face_color="#f87171", edge_color="#991b1b") # Red P5
    draw_3d_box(ax, 2.0, 1.2, 0.8, 0.3, 0.3, face_color="#facc15", edge_color="#854d0e") # Yellow P4
    draw_3d_box(ax, 2.0, 1.5, 0.8, 0.3, 0.3, face_color="#86efac", edge_color="#166534") # Green P3
    draw_3d_box(ax, 2.0, 1.8, 0.8, 0.3, 0.3, face_color="#bfdbfe", edge_color="#1e40af") # Blue P2
    ax.text(3.2, 1.2, "56 × 56 × 40\n28 × 28 × 112\n14 × 14 × 192\n7 × 7 × 1280", fontsize=7.5, color="#1e293b")

    # FPN Block
    draw_3d_box(ax, 4.9, 2.7, 0.6, 1.4, 0.5, face_color="#bfdbfe", edge_color="#2563eb")
    ax.text(5.2, 3.4, "FPN", fontsize=10, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(4.9, 3.4), xytext=(4.5, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # FPN levels list
    for idx, (lbl, col) in enumerate([("P2", "#bfdbfe"), ("P3", "#86efac"), ("P4", "#facc15"), ("P5", "#f87171")]):
        ax.add_patch(patches.Rectangle((4.9, 2.2 - idx * 0.3), 0.3, 0.25, facecolor=col, edgecolor="#334155"))
        ax.text(5.05, 2.32 - idx * 0.3, lbl, fontsize=7, ha="center", va="center", fontweight="bold")
    ax.text(6.0, 1.9, "Pyramid\nFeatures", fontsize=8.5, color="#334155")

    # RPN Block
    draw_3d_box(ax, 6.0, 2.6, 0.9, 1.6, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(6.45, 3.4, "RPN", fontsize=11, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(6.0, 3.4), xytext=(5.6, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # RoI Align Block
    draw_3d_box(ax, 7.3, 2.6, 0.9, 1.6, 0.5, face_color="#f1f5f9", edge_color="#475569")
    ax.text(7.75, 3.5, "RoI\nAlign", fontsize=10.5, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(7.3, 3.4), xytext=(7.0, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))

    # Output head arrow down
    ax.annotate("", xy=(7.75, 1.7), xytext=(7.75, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color="#1e293b"))
    ax.add_patch(patches.Rectangle((7.2, 0.5), 1.3, 1.1, facecolor="#ffffff", edgecolor="#0f172a", lw=1.5))
    ax.text(7.85, 0.2, "Boxes, Labels, Scores", fontsize=9, fontweight="bold", ha="center")

    # Final Output Block right
    draw_3d_box(ax, 8.8, 2.6, 0.25, 1.6, 0.4, face_color="#e2e8f0", edge_color="#475569")
    ax.annotate("", xy=(8.8, 3.4), xytext=(8.4, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))
    ax.annotate("", xy=(8.9, 2.1), xytext=(8.9, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e293b"))
    ax.text(8.9, 1.8, "Output", fontsize=10, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Rendered {out_path}")


if __name__ == "__main__":
    render_fig1_mobilenetv3_block()
    render_fig2_ssdlite_architecture()
    render_fig3_efficientnet_backbone()
    render_fig4_fasterrcnn_resnet50()
    render_fig5_fasterrcnn_efficientnetb0()
