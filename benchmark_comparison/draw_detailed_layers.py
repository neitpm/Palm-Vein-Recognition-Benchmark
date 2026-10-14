# -*- coding: utf-8 -*-
"""
Render a publication-quality, 3D/Isometric deep learning layer diagram
illustrating the exact tensor dimensions, blocks, and connections:
1. SSDLite320 + MobileNetV3-Large (Feature Pyramid & Detection Heads)
2. Faster R-CNN + ResNet-50-FPN (RPN, RoIAlign, Box Predictor)
3. PalmVeinNet (EfficientNet-B0 + ArcFace Metric Head)
Output: benchmark_comparison/detailed_model_layers_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_deep_learning_layers(out_path="benchmark_comparison/detailed_model_layers_diagram.png"):
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Colors
    c_input = "#e0f2fe"       # Light blue
    c_conv = "#dbeafe"        # Soft blue
    c_bneck = "#dcfce7"       # Soft green (Bottleneck / Inverted Residual)
    c_res = "#ede9fe"         # Soft purple (Residual block)
    c_rpn = "#ffedd5"         # Soft orange (RPN)
    c_head = "#fce7f3"        # Soft pink (Detection head / ArcFace)
    c_tensor = "#f8fafc"      # Neutral white
    
    c_border_conv = "#3b82f6"
    c_border_bneck = "#10b981"
    c_border_res = "#8b5cf6"
    c_border_rpn = "#f97316"
    c_border_head = "#ec4899"

    # Header
    ax.text(8.0, 11.6, "Deep Neural Network Layer Architectures & Tensor Flows", 
            fontsize=17, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.0, 11.2, "Precise tensor dimensions, convolutional blocks, feature maps, and loss heads", 
            fontsize=11, color="#64748b", ha="center")

    # =========================================================================
    # SECTION 1: SSDLite320 with MobileNetV3-Large Backbone
    # =========================================================================
    ax.add_patch(patches.FancyBboxPatch((0.5, 7.3), 15.0, 3.6, boxstyle="round,pad=0.2", 
                                        facecolor="#f0fdf4", edgecolor=c_border_bneck, linewidth=2.0))
    ax.text(1.0, 10.6, "1. SSDLite320 + MobileNetV3-Large Layer Architecture", 
            fontsize=12, fontweight="bold", color="#047857")

    # 1.1 Input Tensor (3D-like box)
    ax.add_patch(patches.Rectangle((1.0, 8.0), 1.5, 2.0, facecolor=c_input, edgecolor="#0284c7", lw=1.5))
    ax.text(1.75, 9.1, "Input Tensor\n320x320x3\n(RGB/NIR)", fontsize=9, fontweight="bold", ha="center", va="center")

    ax.annotate("", xy=(3.0, 9.0), xytext=(2.5, 9.0), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 1.2 MobileNetV3 Inverted Residuals Stack
    ax.add_patch(patches.Rectangle((3.0, 7.7), 3.2, 2.6, facecolor=c_bneck, edgecolor=c_border_bneck, lw=1.5))
    ax.text(4.6, 10.0, "MobileNetV3-Large Backbone", fontsize=10, fontweight="bold", color="#065f46", ha="center")
    ax.text(4.6, 8.8, "• Stem: Conv2d 3x3 (s=2) -> 160x160x16\n• 15x Bneck Blocks (Depthwise 3x3/5x5)\n• SE Modules (Squeeze-and-Excitation)\n• Feature Map C4: 20x20x112\n• Feature Map C5: 10x10x960", 
            fontsize=8.2, color="#1f2937", ha="center", va="center")

    ax.annotate("", xy=(6.7, 9.0), xytext=(6.2, 9.0), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 1.3 Multi-scale Extra Feature Pyramid
    ax.add_patch(patches.Rectangle((6.7, 7.7), 3.4, 2.6, facecolor=c_conv, edgecolor=c_border_conv, lw=1.5))
    ax.text(8.4, 10.0, "SSDLite Multi-Scale Feature Maps", fontsize=10, fontweight="bold", color="#1e40af", ha="center")
    ax.text(8.4, 8.8, "• P1: 20x20x112 (from C4)\n• P2: 10x10x960 (from C5)\n• P3: 5x5x512 (Conv 1x1 + 3x3 s=2)\n• P4: 3x3x256 (Conv 1x1 + 3x3 s=2)\n• P5: 2x2x256 (Conv 1x1 + 3x3 s=2)\n• P6: 1x1x128 (Conv 1x1 + 3x3 s=2)", 
            fontsize=8.0, color="#1f2937", ha="center", va="center")

    ax.annotate("", xy=(10.6, 9.0), xytext=(10.1, 9.0), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 1.4 Depthwise Separable Heads & NMS
    ax.add_patch(patches.Rectangle((10.6, 7.7), 4.4, 2.6, facecolor=c_head, edgecolor=c_border_head, lw=1.5))
    ax.text(12.8, 10.0, "SSDLite Heads & Prediction", fontsize=10, fontweight="bold", color="#9d174d", ha="center")
    ax.text(12.8, 8.8, "• Cls Head: Depthwise 3x3 -> Pointwise (num_anchors x 2)\n• Box Head: Depthwise 3x3 -> Pointwise (num_anchors x 4)\n• Default Anchors (Aspect Ratios: 1:1, 1:2, 2:1, 1:3, 3:1)\n• Total Anchors: 3,234 priors\n• Post-Process: Soft-NMS (IoU=0.45) -> 1 BBox ROI", 
            fontsize=8.0, color="#1f2937", ha="center", va="center")

    # =========================================================================
    # SECTION 2: Faster R-CNN with ResNet-50-FPN Backbone
    # =========================================================================
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.4), 15.0, 3.6, boxstyle="round,pad=0.2", 
                                        facecolor="#eff6ff", edgecolor=c_border_conv, lw=2.0))
    ax.text(1.0, 6.7, "2. Faster R-CNN + ResNet-50-FPN Layer Architecture", 
            fontsize=12, fontweight="bold", color="#1d4ed8")

    # 2.1 Input Tensor
    ax.add_patch(patches.Rectangle((1.0, 4.1), 1.5, 2.0, facecolor=c_input, edgecolor="#0284c7", lw=1.5))
    ax.text(1.75, 5.2, "Input Tensor\n600~1000px\n(Multi-scale)", fontsize=9, fontweight="bold", ha="center", va="center")

    ax.annotate("", xy=(3.0, 5.1), xytext=(2.5, 5.1), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 2.2 ResNet-50 + FPN
    ax.add_patch(patches.Rectangle((3.0, 3.8), 3.2, 2.6, facecolor=c_res, edgecolor=c_border_res, lw=1.5))
    ax.text(4.6, 6.1, "ResNet-50 + FPN Backbone", fontsize=10, fontweight="bold", color="#6b21a8", ha="center")
    ax.text(4.6, 4.9, "• 50 Residual Conv Layers (Bottlenecks)\n• Bottom-up stages: C2, C3, C4, C5\n• Top-down pathway + Lateral 1x1 Conv\n• FPN Levels: P2, P3, P4, P5, P6\n• Uniform 256 channels across all levels", 
            fontsize=8.2, color="#1f2937", ha="center", va="center")

    ax.annotate("", xy=(6.7, 5.1), xytext=(6.2, 5.1), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 2.3 RPN (Region Proposal Network)
    ax.add_patch(patches.Rectangle((6.7, 3.8), 3.4, 2.6, facecolor=c_rpn, edgecolor=c_border_rpn, lw=1.5))
    ax.text(8.4, 6.1, "Region Proposal Network (RPN)", fontsize=10, fontweight="bold", color="#c2410c", ha="center")
    ax.text(8.4, 4.9, "• 3x3 Conv (256-d) on all FPN maps\n• Objectness Head (1x1 Conv -> 2 logits)\n• RPN BBox Head (1x1 Conv -> 4 deltas)\n• Generates ~1000 candidate proposals\n• Proposal NMS filtering -> Top 300 RoIs", 
            fontsize=8.0, color="#1f2937", ha="center", va="center")

    ax.annotate("", xy=(10.6, 5.1), xytext=(10.1, 5.1), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 2.4 RoIAlign & Two-Stage Box Head
    ax.add_patch(patches.Rectangle((10.6, 3.8), 4.4, 2.6, facecolor=c_head, edgecolor=c_border_head, lw=1.5))
    ax.text(12.8, 6.1, "RoIAlign & Fast R-CNN Head", fontsize=10, fontweight="bold", color="#9d174d", ha="center")
    ax.text(12.8, 4.9, "• RoIAlign: 7x7 Bilinear sampling (prevents quantization)\n• 2x FC Layers (1024-d each)\n• Box Classifier (FC -> 2 classes: bg / palm)\n• Box Regressor (FC -> 4 coordinates)\n• High latency & proposal redundancy for single class", 
            fontsize=8.0, color="#1f2937", ha="center", va="center")

    # =========================================================================
    # SECTION 3: PalmVeinNet (Deep Metric Learning / ArcFace)
    # =========================================================================
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.4), 15.0, 2.7, boxstyle="round,pad=0.2", 
                                        facecolor="#faf5ff", edgecolor=c_border_res, lw=2.0))
    ax.text(1.0, 2.8, "3. PalmVeinNet (EfficientNet-B0 + ArcFace) Metric Recognition Layer", 
            fontsize=12, fontweight="bold", color="#6b21a8")

    # 3.1 Cropped Enhanced ROI
    ax.add_patch(patches.Rectangle((1.0, 0.7), 2.2, 1.8, facecolor=c_input, edgecolor="#0284c7", lw=1.5))
    ax.text(2.1, 1.8, "Enhanced Vein ROI", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(2.1, 1.2, "224x224x3\nCLAHE Enhanced\nGaussian Denoised", fontsize=8.5, ha="center", color="#444444")

    ax.annotate("", xy=(3.8, 1.6), xytext=(3.3, 1.6), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 3.2 EfficientNet-B0 Backbone
    ax.add_patch(patches.Rectangle((3.8, 0.7), 3.2, 1.8, facecolor=c_bneck, edgecolor=c_border_bneck, lw=1.5))
    ax.text(5.4, 2.0, "EfficientNet-B0 Backbone", fontsize=9.5, fontweight="bold", color="#065f46", ha="center")
    ax.text(5.4, 1.3, "• MBConv1 & MBConv6 Blocks\n• Squeeze-and-Excitation ratio=0.25\n• Global Average Pooling -> 1280-d", fontsize=8.2, color="#1f2937", ha="center")

    ax.annotate("", xy=(7.6, 1.6), xytext=(7.1, 1.6), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 3.3 Embedding Projection
    ax.add_patch(patches.Rectangle((7.6, 0.7), 3.2, 1.8, facecolor=c_conv, edgecolor=c_border_conv, lw=1.5))
    ax.text(9.2, 2.0, "Embedding Head", fontsize=9.5, fontweight="bold", color="#1e40af", ha="center")
    ax.text(9.2, 1.3, "• BatchNorm1d + Dropout (0.2)\n• FC Layer (1280 -> 512)\n• L2-Normalization (||e||_2 = 1)", fontsize=8.2, color="#1f2937", ha="center")

    ax.annotate("", xy=(11.4, 1.6), xytext=(10.9, 1.6), arrowprops=dict(arrowstyle="-|>", lw=2, color="#475569"))

    # 3.4 ArcFace Loss / Cosine Matcher
    ax.add_patch(patches.Rectangle((11.4, 0.7), 3.6, 1.8, facecolor=c_head, edgecolor=c_border_head, lw=1.5))
    ax.text(13.2, 2.0, "ArcFace Loss & Matcher", fontsize=9.5, fontweight="bold", color="#9d174d", ha="center")
    ax.text(13.2, 1.3, "• Angular Margin: m=0.5, Scale s=30\n• Hyperspherical Geodesic Separation\n• Cosine Distance for 1:1 & 1:N Match", fontsize=8.2, color="#1f2937", ha="center")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Rendered detailed model layers diagram at {out_path}")

if __name__ == "__main__":
    draw_deep_learning_layers()
