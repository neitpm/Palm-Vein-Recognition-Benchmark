# -*- coding: utf-8 -*-
"""
Script so sánh định lượng (Quantitative Comparison) và vẽ biểu đồ đối chuẩn
giữa SSDLite-MobileNetV3 và Faster R-CNN (ResNet-50 / EfficientNet-B0).
Xuất ảnh: benchmark_comparison/model_comparison_radar.png
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_comparison():
    labels = ['mAP@0.5:0.95', 'mAP@0.3', 'Precision', 'Recall', 'F1-Score']
    ssdlite_scores = [0.8508, 0.9896, 0.9994, 0.9994, 0.9994]
    fasterrcnn_scores = [0.8000, 1.0000, 0.4167, 0.5000, 0.4545]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - width/2, ssdlite_scores, width, label='SSDLite (MobileNetV3-Large)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, fasterrcnn_scores, width, label='Faster R-CNN (ResNet-50)', color='#1f77b4')

    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    ax.set_title('Detection & Biometric ROI Performance: SSDLite vs. Faster R-CNN', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Thêm giá trị số trên từng cột
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    out_file = Path("benchmark_comparison/model_comparison_chart.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[OK] Saved comparison chart to {out_file}")

if __name__ == "__main__":
    plot_comparison()
