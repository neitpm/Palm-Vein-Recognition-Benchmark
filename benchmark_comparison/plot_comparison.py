# -*- coding: utf-8 -*-
"""
Render a clean, high-contrast, modern graphic comparison chart.
Output: benchmark_comparison/model_comparison_chart.png
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_modern_comparison():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    labels = ['mAP@0.5:0.95', 'mAP@0.3', 'Precision', 'Recall', 'F1-Score']
    ssdlite_scores = [0.8508, 0.9896, 0.9994, 0.9994, 0.9994]
    fasterrcnn_scores = [0.8000, 1.0000, 0.4167, 0.5000, 0.4545]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300, facecolor='#ffffff')
    ax.set_facecolor('#fafbfc')

    # Colors: vibrant, distinct, and pleasant
    c_ssd = '#10b981'    # Emerald green
    c_rcnn = '#3b82f6'   # Bright royal blue

    rects1 = ax.bar(x - width/2, ssdlite_scores, width, label='SSDLite (MobileNetV3-Large)', 
                    color=c_ssd, edgecolor='#059669', linewidth=1.5, zorder=3)
    rects2 = ax.bar(x + width/2, fasterrcnn_scores, width, label='Faster R-CNN (ResNet-50-FPN)', 
                    color=c_rcnn, edgecolor='#2563eb', linewidth=1.5, zorder=3)

    # Grid & Spines
    ax.grid(axis='y', linestyle='--', alpha=0.6, color='#cbd5e1', zorder=0)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#94a3b8')
        ax.spines[spine].set_linewidth(1.0)

    # Typography
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_title('Quantitative Benchmark: SSDLite vs. Faster R-CNN on Palm ROI', 
                 fontsize=15, fontweight='bold', color='#0f172a', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold', color='#334155')
    ax.set_ylim(0, 1.18)

    # Legend
    legend = ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', 
                       fontsize=11, loc='upper right', framealpha=0.95)
    for text in legend.get_texts():
        text.set_color('#1e293b')

    # Data value callout badges
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#047857')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1d4ed8')

    out_file = Path("benchmark_comparison/model_comparison_chart.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Re-generated polished comparison chart at {out_file}")

if __name__ == "__main__":
    plot_modern_comparison()
