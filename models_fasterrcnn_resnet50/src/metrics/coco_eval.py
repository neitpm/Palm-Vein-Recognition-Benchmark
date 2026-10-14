import numpy as np, torch
from pycocotools.cocoeval import COCOeval

@torch.no_grad()
def coco_map(model, data_loader, coco_gt, iou_type="bbox", device="cuda",
             conf_thres=0.30, label2cat=None, debug_show_first=5):
    model.eval()
    results, img_ids = [], []

    with torch.no_grad():
        for imgs, tgts in data_loader:
            # Ép float [0,1] nếu còn uint8
            imgs = [ (i.float().div(255) if i.dtype == torch.uint8 else i).to(device) for i in imgs ]
            outs = model(imgs)

            for i, out in enumerate(outs):
                image_id = int(tgts[i]["image_id"])
                img_ids.append(image_id)
                if len(out["boxes"]) == 0:
                    continue

                orig_w = coco_gt.imgs[image_id]['width']
                orig_h = coco_gt.imgs[image_id]['height']
                in_h, in_w = imgs[i].shape[-2], imgs[i].shape[-1]

                boxes  = out["boxes"].detach().cpu().clone()
                scores = out["scores"].detach().cpu()
                labels = out["labels"].detach().cpu()

                keep = scores > conf_thres
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if len(boxes) == 0: 
                    continue

                sx = float(orig_w) / float(in_w)
                sy = float(orig_h) / float(in_h)
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy

                boxes_xywh = boxes.clone()
                boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

                for box, score, label in zip(boxes_xywh, scores, labels):
                    l = int(label)
                    if l == 0: 
                        continue
                    if (label2cat is None) or (l not in label2cat): 
                        continue
                    cat_id = int(label2cat[l])
                    x, y, w, h = [float(v) for v in box.tolist()]
                    if w <= 0 or h <= 0:
                        continue
                    results.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [x, y, w, h],
                        "score": float(score),
                    })

    if len(results) == 0:
        print("[WARN] coco_map: không có detections nào vượt ngưỡng conf_thres.")
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)
    eval_std = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    eval_std.params.imgIds = sorted(list(set(img_ids)))
    eval_std.evaluate()
    eval_std.accumulate()
    eval_std.summarize()
    map_std = float(eval_std.stats[0])

    eval_03 = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    eval_03.params.imgIds = sorted(list(set(img_ids)))
    eval_03.params.iouThrs = np.array([0.30])
    eval_03.evaluate()
    eval_03.accumulate()
    eval_03.summarize()
    map_03 = float(eval_03.stats[0])

    return map_std, map_03
