import sys, pathlib, yaml, warnings, shutil, torch, json, matplotlib.pyplot as plt, os, collections, random
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# project
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.utils.common import seed_everything, get_device, amp_ctx
from src.dataset.coco_palm import COCODataset, collate
from src.metrics.coco_eval import coco_map
from src.metrics.prf import detection_prf_cm

PROJECT_ROOT = ROOT

def get_transforms_faster(train: bool):
    aug = []
    if train:
        aug += [A.HorizontalFlip(p=0.5),
                A.ColorJitter(0.3,0.3,0.3,0.3,p=0.5)]
    aug += [ToTensorV2()]
    return A.Compose(aug, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["cls"]))

def load_cfg(path="configs/fasterrcnn_resnet50_palm.yaml"):
    p = Path(path)
    if not p.is_absolute(): p = PROJECT_ROOT / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = cfg.get("root_path","")
    for k in ["train_ann","val_ann","test_ann"]:
        if isinstance(cfg.get(k), str):
            cfg[k] = cfg[k].replace("${root_path}", root)
    return cfg

def build_model(num_classes: int, min_size=None, max_size=None, trainable_backbone_layers=3, freeze_backbone=False):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    kwargs = {"weights": weights, "trainable_backbone_layers": trainable_backbone_layers}
    if min_size is not None: kwargs["min_size"] = int(min_size)
    if max_size is not None: kwargs["max_size"] = int(max_size)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)
    if freeze_backbone:
        for name, param in model.backbone.body.named_parameters():
            param.requires_grad = False
    # replace predictor
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    return model

def print_dataset_diagnostics(name, ds, diag_dir, sample_n=8):
    """
    In diagnostics lên console và lưu JSON vào diag_dir/{name}_diagnostics.json
    """
    prefix = f"[{name}]"
    coco = ds.coco_ds.coco
    used_cat_ids = sorted({a['category_id'] for a in coco.anns.values()})
    print(f"{prefix} used_cat_ids in annotations: {used_cat_ids}")
    try:
        print(f"{prefix} ds.cat2label: {ds.cat2label}")
        print(f"{prefix} ds.label2cat: {ds.label2cat}")
        print(f"{prefix} num_classes: {ds.num_classes}")
    except Exception:
        pass

    # bbox count by category_id
    bbox_count = collections.Counter()
    for ann in coco.anns.values():
        bbox_count[int(ann['category_id'])] += 1
    bbox_count_dict = dict(sorted(bbox_count.items()))
    print(f"{prefix} bbox count by category_id: {bbox_count_dict}")

    # sample label distribution after transform:
    label_counter = collections.Counter()
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    taken = 0
    for idx in idxs:
        if taken >= sample_n:
            break
        try:
            img, tgt = ds[idx]   # áp transform (nếu có)
            labs = tgt.get("labels", None)
            if labs is None:
                taken += 1
                continue
            if isinstance(labs, torch.Tensor):
                labs = labs.cpu().numpy().tolist()
            if isinstance(labs, (int, float)):
                label_counter[int(labs)] += 1
            else:
                for l in labs:
                    label_counter[int(l)] += 1
            taken += 1
        except Exception:
            continue

    sample_dist = dict(sorted(label_counter.items()))
    print(f"{prefix} sample label distribution (after transform): {sample_dist}")
    print()

    # Save diagnostics JSON
    diag_obj = {
        "used_cat_ids": used_cat_ids,
        "cat2label": getattr(ds, "cat2label", None),
        "label2cat": getattr(ds, "label2cat", None),
        "num_classes": getattr(ds, "num_classes", None),
        "bbox_count": bbox_count_dict,
        "sample_label_dist": sample_dist,
    }
    diag_path = diag_dir / f"{name}_diagnostics.json"
    try:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ Không thể lưu diagnostics:", e)
    # after saving diag_obj as JSON (diag_path)
    txt_path = diag_dir / f"{name}_diagnostics.txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"[{name}] used_cat_ids: {diag_obj['used_cat_ids']}\n")
            f.write(f"[{name}] cat2label: {diag_obj['cat2label']}\n")
            f.write(f"[{name}] label2cat: {diag_obj['label2cat']}\n")
            f.write(f"[{name}] num_classes: {diag_obj['num_classes']}\n")
            f.write(f"[{name}] bbox_count: {diag_obj['bbox_count']}\n")
            f.write(f"[{name}] sample_label_dist: {diag_obj['sample_label_dist']}\n")
    except Exception as e:
        print("⚠️ Không thể ghi diagnostics TXT:", e)


def plot_and_save_curves(plots_dir, train_losses, val_losses, maps, maps03, precs, recs, f1s):
    epochs = list(range(1, len(train_losses)+1))
    if len(epochs) > 0:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(epochs, train_losses, marker='o', label='train_loss')
        ax.plot(epochs, val_losses, marker='o', label='val_loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Loss curve'); ax.grid(True); ax.legend()
        fig.savefig(plots_dir / "loss_curve.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(epochs, maps, marker='o', label='mAP@0.5:0.95')
        ax.plot(epochs, maps03, marker='o', label='mAP@0.3')
        ax.plot(epochs, f1s, marker='o', label='F1 (macro)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
        ax.set_title('mAP / F1 curve'); ax.grid(True); ax.legend()
        fig.savefig(plots_dir / "metrics_curve.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def main():
    cfg = load_cfg()
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    use_amp = (device.type == "cuda")
    print("Device:", device)

    # Dataset
    train_set = COCODataset(f"{cfg['root_path']}/train", cfg["train_ann"], get_transforms_faster(train=True))
    val_set   = COCODataset(f"{cfg['root_path']}/valid", cfg["val_ann"],  get_transforms_faster(train=False))

    # Setup structured output folders
    save_dir = PROJECT_ROOT / cfg.get("save_dir", "output")
    checkpoints_dir = save_dir / "checkpoints"
    plots_dir       = save_dir / "plots"
    logs_dir        = save_dir / "logs"
    diag_dir        = save_dir / "diagnostics"
    for d in (save_dir, checkpoints_dir, plots_dir, logs_dir, diag_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Print & save diagnostics
    print_dataset_diagnostics("train", train_set, diag_dir, sample_n=8)
    print_dataset_diagnostics("valid", val_set, diag_dir, sample_n=8)

    # Compare category ids
    train_cat_ids = sorted({a['category_id'] for a in train_set.coco_ds.coco.anns.values()})
    valid_cat_ids = sorted({a['category_id'] for a in val_set.coco_ds.coco.anns.values()})
    print(f"[compare] train used_cat_ids: {train_cat_ids}")
    print(f"[compare] valid used_cat_ids: {valid_cat_ids}")
    if train_cat_ids == valid_cat_ids:
        print("✓ OK: category_id train và val giống nhau.")
    else:
        print("⚠️ Warning: category_id train và val khác nhau!")

    train_loader = DataLoader(train_set, batch_size=cfg["batch_train"], shuffle=True,
                              collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    val_loader   = DataLoader(val_set,   batch_size=cfg["batch_val"], shuffle=False,
                              collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

    # Model/Optim/Sched
    model = build_model(num_classes=train_set.num_classes,
                    min_size=cfg.get("min_size"),
                    max_size=cfg.get("max_size"),
                    trainable_backbone_layers=0,   # hoặc 1-3
                    freeze_backbone=False).to(device)

    opt = torch.optim.SGD(model.parameters(),
                          lr=cfg["lr"],
                          momentum=cfg.get("momentum", 0.9),
                          weight_decay=cfg.get("weight_decay", 5e-5))
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["step_size"], gamma=cfg["gamma"])

    coco_gt_val = val_set.coco_ds.coco
    gt_ids = sorted({a["category_id"] for a in coco_gt_val.anns.values()})
    label2cat = {1: gt_ids[0]}  # map label=1 -> đúng category_id trong val

    best_map = -1.0
    best_loss = float("inf")
    # Early stopping params (monitor val_loss only)
    early_stop = {
        "enabled": True,        # bật early stopping
        "monitor": "val_loss",  # 'val_loss' (lower is better)
        "patience": 5,          # dừng nếu val_loss không giảm trong 5 epoch liên tiếp
        "min_delta": 1e-4,      # phải giảm ít nhất min_delta mới tính là 'cải thiện'
    }
    _no_improve_count = 0
    # Khởi tạo giá trị best monitor: dùng best_loss (lower better)
    _best_monitor_value = float(best_loss)

    # arrays for logging
    train_losses = []
    val_losses = []
    maps = []
    maps03 = []
    precs = []
    recs = []
    f1s = []

    # ensure logs file exists (start fresh)
    metrics_file = logs_dir / "metrics_epoch.json"
    if metrics_file.is_file():
        metrics_file.unlink()
    all_metrics = []

    for ep in range(cfg["epochs"]):
        # -------- TRAIN --------
        model.train()
        total = 0.0
        for imgs, tgts in tqdm(train_loader, desc=f"Train {ep+1}/{cfg['epochs']}"):
            # ensure float tensors in [0,1] for torchvision model
            imgs = [ (i.float().div(255) if i.dtype == torch.uint8 else i).to(device) for i in imgs ]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            with amp_ctx(use_amp):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))
        train_losses.append(train_loss)

        # -------- VAL LOSS --------
        was_training = model.training
        model.train()
        vtot = 0.0
        with torch.no_grad():
            for imgs, tgts in tqdm(val_loader, desc=f"ValLoss {ep+1}", leave=False):
                imgs = [ (i.float().div(255) if i.dtype == torch.uint8 else i).to(device) for i in imgs ]
                tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
                ld = model(imgs, tgts)   # now returns dict of losses
                # ld is a dict of tensors; sum and accumulate
                vtot += sum(ld.values()).item()
        val_loss = vtot / max(1, len(val_loader))
        val_losses.append(val_loss)
        # restore previous training/eval state
        if not was_training:
            model.eval()

        # -------- METRICS (COCO mAP + P/R/F1) --------
        mAP, mAP03 = coco_map(model, val_loader, coco_gt=coco_gt_val,
                              label2cat=label2cat, conf_thres=cfg["conf_eval"], device=device)
        prec, rec, f1, _ = detection_prf_cm(model, val_loader, train_set.num_classes,
                                            score_thr=cfg["conf_eval"], device=device)

        maps.append(mAP); maps03.append(mAP03)
        precs.append(prec); recs.append(rec); f1s.append(f1)

        print(f"EP{ep+1}: train={train_loss:.4f}  val={val_loss:.4f}  "
              f"mAP@.5:.95={mAP:.4f}  mAP@0.3={mAP03:.4f}  "
              f"P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

        # -------- SAVE BEST (to checkpoints_dir) --------
        if mAP >= best_map:
            best_map = mAP
            torch.save(model.state_dict(), checkpoints_dir/"best_map.pth")
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoints_dir/"best_loss.pth")
        # --- Early stopping check (monitor val_loss only) ---
        if early_stop.get("enabled", False):
            mon = early_stop.get("monitor", "val_loss")
            # We assume mon == "val_loss" as you requested
            min_delta = float(early_stop.get("min_delta", 1e-4))
            current = float(val_loss)  # lower is better
            improved = (current < (_best_monitor_value - min_delta))

            if improved:
                _best_monitor_value = current
                _no_improve_count = 0
                print(f"[early-stopping] val_loss improved -> {_best_monitor_value:.6f}. reset no_improve_count.")
            else:
                _no_improve_count += 1
                print(f"[early-stopping] val_loss not improved. current={current:.6f} best={_best_monitor_value:.6f} no_improve={_no_improve_count}/{early_stop['patience']}")

            if _no_improve_count >= int(early_stop.get("patience", 5)):
                print(f"⛔ Early stopping triggered: val_loss did not decrease for {_no_improve_count} epochs (patience={early_stop['patience']}). Stopping training.")
                # Save last checkpoint
                try:
                    last_ckpt = checkpoints_dir / f"last_checkpoint_ep{ep+1}.pth"
                    torch.save(model.state_dict(), last_ckpt)
                    print(f"💾 Saved last checkpoint: {last_ckpt}")
                except Exception as e:
                    print("⚠️ Không thể lưu last checkpoint:", e)
                # Write early-stop note into train_log.txt for record
                try:
                    with open(logs_dir / "train_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"EARLY_STOPPING at epoch {ep+1}: val_loss did not improve for {_no_improve_count} epochs (patience={early_stop['patience']}).\\n")
                except Exception:
                    pass
                break

        sch.step()

        # Persist incremental metrics
        epoch_metrics = {
            "epoch": ep+1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "map_0.5_0.95": float(mAP),
            "map_0.3": float(mAP03),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        all_metrics.append(epoch_metrics)
        try:
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("⚠️ Lỗi khi ghi metrics_epoch.json:", e)
        # --- Append human-readable line to train_log.txt ---
        train_log_path = logs_dir / "train_log.txt"
        line = (f"EP{ep+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"mAP={mAP:.4f} | mAP@0.3={mAP03:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f}\n")
        try:
            with open(train_log_path, "a", encoding="utf-8") as f:
                # if file is new, write header first
                if train_log_path.stat().st_size == 0:
                    f.write("epoch | train_loss | val_loss | mAP | mAP@0.3 | precision | recall | f1\n")
                f.write(line)
        except Exception as e:
            print("⚠️ Không thể ghi train_log.txt:", e)

    # Đồng bộ best_model.pth theo policy (keep in checkpoints)
    policy = str(cfg.get("best_select","map")).lower()
    src = checkpoints_dir/("best_map.pth" if policy=="map" else "best_loss.pth")
    if src.is_file():
        shutil.copyfile(src, checkpoints_dir/"best_model.pth")
        print(f"✓ synced best_model.pth -> {src.name}")

    # Sau khi huấn luyện xong: vẽ và lưu biểu đồ vào plots_dir
    try:
        plot_and_save_curves(plots_dir, train_losses, val_losses, maps, maps03, precs, recs, f1s)
        print(f"✓ Saved plots to: {plots_dir}")
    except Exception as e:
        print("⚠️ Lỗi khi vẽ/lưu biểu đồ:", e)

    # Lưu final metrics summary vào logs_dir
    summary = {
        "best_map": best_map,
        "best_loss": best_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "maps": maps,
        "maps03": maps03,
        "precision": precs,
        "recall": recs,
        "f1": f1s
    }
    try:
        with open(logs_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ Lỗi khi ghi metrics_summary.json:", e)

    print("✓ Training finished. Artifacts:")
    print("  - checkpoints:", sorted(x.name for x in checkpoints_dir.iterdir()))
    print("  - plots:", sorted(x.name for x in plots_dir.iterdir()))
    print("  - logs:", sorted(x.name for x in logs_dir.iterdir()))
    print("  - diagnostics:", sorted(x.name for x in diag_dir.iterdir()))

if __name__ == "__main__":
    main()
