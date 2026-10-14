# -*- coding: utf-8 -*-
"""
Đánh giá mô hình SSDLite320-MobileNetV3:
- mAP@0.5:0.95 (COCOeval) + mAP@0.3
- Precision / Recall / F1 (macro)
- Confusion Matrix (lưu PNG)
- (tuỳ chọn) quét ngưỡng confidence để tối ưu F1
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # D:\PMT_Paper

if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import os, json, yaml, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, json, argparse, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from torch.utils.data import DataLoader

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import yaml
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

# bootstrap sys.path
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.coco_palm import COCODataset, collate, get_transforms
from src.metrics.coco_eval import coco_map
from src.metrics.prf import detection_prf_cm
from src.utils.common import get_device, seed_everything


def load_cfg(path):
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy config: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    # expand ${root_path}
    root = cfg.get("root_path", "")
    for k in ["train_ann", "val_ann", "test_ann"]:
        if isinstance(cfg.get(k), str):
            cfg[k] = cfg[k].replace("${root_path}", root)
    return cfg


def build_loader(cfg, split, batch, workers, pin_memory=True):
    if split == "valid":
        img_root = Path(cfg["root_path"]) / "valid"
        ann_file = cfg["val_ann"]
        tf = get_transforms(train=False)
    # elif split == "test":
    #     img_root = Path(cfg["root_path"]) / "test"
    #     ann_file = cfg["test_ann"]
    #     tf = get_transforms(train=False)
    else:
        img_root = Path(cfg["root_path"]) / "train"
        ann_file = cfg["train_ann"]
        tf = get_transforms(train=False)
    ds = COCODataset(str(img_root), ann_file, tf)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        collate_fn=collate, num_workers=workers,
                        pin_memory=pin_memory)
    return ds, loader


def plot_confusion_matrix(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Pred")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sweep_confidence(model, loader, num_classes, device, conf_list):
    # Trả về dict {conf: (prec, rec, f1)}
    out = {}
    for c in conf_list:
        prec, rec, f1, _ = detection_prf_cm(model, loader, num_classes,
                                            iou_thr=0.5, score_thr=c, device=device)
        out[float(c)] = (float(prec), float(rec), float(f1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/ssd_palm.yaml")
    ap.add_argument("--weights", default="output/best_model.pth", help="đường dẫn best model")
    ap.add_argument("--split", default="test", choices=["valid", "test"], help="tập đánh giá")
    ap.add_argument("--conf", type=float, default=0.30, help="ngưỡng confidence dùng cho PR/F1")
    ap.add_argument("--sweep", action="store_true", help="quét nhiều ngưỡng confidence")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    print("Device:", device)

    # loader
    ds, loader = build_loader(cfg, split=args.split,
                              batch=cfg["batch_val"],
                              workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])

    coco_gt = ds.coco_ds.coco
    gt_cat_ids = sorted({a['category_id'] for a in coco_gt.anns.values()})
    label2cat = {1: gt_cat_ids[0]}
    print(f"GT category_ids ({args.split}):", gt_cat_ids)

    # model
    num_classes = ds.num_classes  # 2 (bg + palm)
    model = ssdlite320_mobilenet_v3_large(weights_backbone="DEFAULT", num_classes=num_classes).to(device)
    save_dir = PROJECT_ROOT / "output"
    save_dir.mkdir(parents=True, exist_ok=True)
    # khi lưu:
    torch.save(model.state_dict(), save_dir / "best_model.pth")
    ckpt = Path(args.weights)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"Không thấy weights: {ckpt}")
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()

    # mAP COCO
    map_std, map_03 = coco_map(model, loader, coco_gt=coco_gt, label2cat=label2cat,
                               conf_thres=args.conf, device=device)
    print(f"[{args.split}] mAP@0.5:0.95 = {map_std:.4f} | mAP@0.3 = {map_03:.4f}")

    # P/R/F1 + Confusion Matrix
    prec, rec, f1, cm = detection_prf_cm(model, loader, num_classes, iou_thr=0.5,
                                         score_thr=args.conf, device=device)
    print(f"[{args.split}] Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # Lưu kết quả
    out_dir = PROJECT_ROOT / "output" / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, ["bg(0)", "palm(1)"], out_dir / "confusion_matrix.png")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "conf_thr": args.conf,
            "map_0.5_0.95": map_std,
            "map_0.3": map_03,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cm": cm.tolist(),
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Đã lưu: {out_dir/'metrics.json'} và confusion_matrix.png")

    # Sweep confidence (tuỳ chọn)
    if args.sweep:
        confs = np.linspace(0.05, 0.9, 18)
        table = sweep_confidence(model, loader, num_classes, device, confs)
        with open(out_dir / "conf_sweep.json", "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã lưu: {out_dir/'conf_sweep.json'} (chọn ngưỡng tốt nhất theo F1)")


if __name__ == "__main__":
    main()
