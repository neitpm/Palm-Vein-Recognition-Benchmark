import numpy as np, torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def compute_iou_batch(box, boxes):
    if len(boxes) == 0:
        return np.array([])
    boxes = np.array(boxes)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

@torch.no_grad()
def detection_prf_cm(model, loader, num_classes, iou_thr=0.5, score_thr=0.30, device="cuda"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, tgts in loader:
            # Ép float [0,1] nếu còn uint8
            imgs = [
                (i.float().div(255) if i.dtype == torch.uint8 else i).to(device)
                for i in imgs
            ]
            outs = model(imgs)
        for out, tgt in zip(outs, tgts):
            gt_boxes = tgt["boxes"].cpu().numpy()
            gt_labels = tgt["labels"].cpu().numpy()
            pred_boxes = out["boxes"].cpu().numpy()
            pred_labels = out["labels"].cpu().numpy()
            pred_scores = out["scores"].cpu().numpy()

            keep = pred_scores >= score_thr
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]

            matched_gt = set()
            for pb, pl in zip(pred_boxes, pred_labels):
                ious = compute_iou_batch(pb, gt_boxes)
                if ious.size == 0:
                    y_true.append(0); y_pred.append(int(pl)); continue
                gi = int(np.argmax(ious))
                if ious[gi] >= iou_thr and (gi not in matched_gt):
                    y_true.append(int(gt_labels[gi])); y_pred.append(int(pl))
                    matched_gt.add(gi)
                else:
                    y_true.append(0); y_pred.append(int(pl))

            for gi, gl in enumerate(gt_labels):
                if gi not in matched_gt:
                    y_true.append(int(gl)); y_pred.append(0)

    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, np.zeros((num_classes, num_classes), dtype=int)

    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return prec, rec, f1, cm
