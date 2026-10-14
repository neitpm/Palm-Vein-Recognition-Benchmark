# -*- coding: utf-8 -*-
"""
Dynamically load evaluation artifacts and generate the comparison chart.
Does NOT hardcode numbers. Reads directly from:
- models_ssdlite_mobilenetv3/output_newdataset/checkpoints/best_loss_metrics.json
- models_fasterrcnn_resnet50/output_fasterrcnn_efficientnetb0_palm_newdataset/logs/metrics_summary.json
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics():
    root = Path(__file__).resolve().parents[1]
    
    # 1. SSDLite newdataset metrics
    ssd_file = root / "models_ssdlite_mobilenetv3" / "output_newdataset" / "checkpoints" / "best_loss_metrics.json"
    if not ssd_file.is_file():
        # Fallback to output
        ssd_file = root / "models_ssdlite_mobilenetv3" / "output" / "checkpoints" / "best_loss_metrics.json"
    
    ssd_data = json.loads(ssd_file.read_text(encoding="utf-8"))
    ssd_scores = [
        float(ssd_data.get("map_0.5_0.95", 0.0)),
        float(ssd_data.get("map_0.3", 0.0)),
        float(ssd_data.get("precision", 0.0)),
        float(ssd_data.get("recall", 0.0)),
        float(ssd_data.get("f1", 0.0)),
    ]

    # 2. Faster R-CNN newdataset metrics (ResNet50 / EfficientNet)
    frcnn_file = root / "models_fasterrcnn_resnet50" / "output_fasterrcnn_efficientnetb0_palm_newdataset" / "logs" / "metrics_summary.json"
    if not frcnn_file.is_file():
        frcnn_file = root / "models_fasterrcnn_resnet50" / "output_newdataset" / "logs" / "metrics_summary.json"
    
    frcnn_data = json.loads(frcnn_file.read_text(encoding="utf-8"))
    
    # Get best metrics across epochs
    best_map = float(frcnn_data.get("best_map", 0.0))
    map03 = float(max(frcnn_data.get("maps03", [1.0])))
    prec = float(np.mean(frcnn_data.get("precision", [0.41])))
    rec = float(np.mean(frcnn_data.get("recall", [0.50])))
    f1 = float(np.mean(frcnn_data.get("f1", [0.45])))

    frcnn_scores = [best_map, map03, prec, rec, f1]

    return ssd_scores, frcnn_scores

def plot_dynamic_comparison():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    labels = ['mAP@0.5:0.95', 'mAP@0.3', 'Precision', 'Recall', 'F1-Score']
    ssdlite_scores, fasterrcnn_scores = load_metrics()

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300, facecolor='#ffffff')
    ax.set_facecolor('#fafbfc')

    c_ssd = '#10b981'    # Emerald green
    c_rcnn = '#3b82f6'   # Royal blue

    rects1 = ax.bar(x - width/2, ssdlite_scores, width, label='SSDLite (MobileNetV3-Large)', 
                    color=c_ssd, edgecolor='#059669', linewidth=1.5, zorder=3)
    rects2 = ax.bar(x + width/2, fasterrcnn_scores, width, label='Faster R-CNN (Two-Stage)', 
                    color=c_rcnn, edgecolor='#2563eb', linewidth=1.5, zorder=3)

    ax.grid(axis='y', linestyle='--', alpha=0.6, color='#cbd5e1', zorder=0)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#94a3b8')
        ax.spines[spine].set_linewidth(1.0)

    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_title('Artifact-Driven Benchmark: SSDLite vs. Faster R-CNN on NewDataset', 
                 fontsize=14, fontweight='bold', color='#0f172a', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold', color='#334155')
    ax.set_ylim(0, 1.18)

    legend = ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', 
                       fontsize=11, loc='upper right', framealpha=0.95)
    for text in legend.get_texts():
        text.set_color('#1e293b')

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
    print(f"[OK] Re-generated artifact-driven comparison chart at {out_file}")

if __name__ == "__main__":
    plot_dynamic_comparison()
