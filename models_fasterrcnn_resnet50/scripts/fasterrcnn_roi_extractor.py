# -*- coding: utf-8 -*-
# Faster R-CNN ROI extractor (for your fasterrcnn_resnet50_palm setup)
# Usage: create extractor = FasterROIExtractor(...); extractor.process_image(...)

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
import cv2, torch, numpy as np, json, warnings
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# import your build_model function from training script location
# adjust import path if needed
from scripts.train_fasterrcnn_resnet50_palm import build_model  # or from your module
# if that import path doesn't work, copy build_model into this file

warnings.filterwarnings("ignore", category=UserWarning)

class FasterROIExtractor:
    def __init__(self, weights=r"D:\PMT_Paper_Fasterrcnn-resnet50\output_newdataset\checkpoints\best_loss.pth",
                 num_classes=2, device=None, resize_wh=(800,1333),
                 expand=0.1, class_names=("bg","palm"), vis_save=False):
        """
        resize_wh: (in_h, in_w) or (height, width) used to create tensor size.
                   Faster R-CNN will still internally resize, but we use a deterministic resize
                   so boxes scale back predictably. Default choosen near model defaults.
        """
        self.project_root = ROOT
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.expand = float(expand)
        self.vis_save = vis_save

        # build model same config as training
        h, w = resize_wh
        # Note: trainable_backbone_layers, min/max_size can be passed if needed
        self.model = build_model(num_classes=num_classes, min_size=h, max_size=w)
        # load checkpoint robustly
        ckpt = Path(weights)
        if not ckpt.is_absolute():
            ckpt = self.project_root / ckpt
        if not ckpt.is_file():
            raise FileNotFoundError(f"Không thấy weights: {ckpt}")

        state = torch.load(str(ckpt), map_location=self.device)
        # if state is a state_dict, try load; if it's full model, accept it
        if isinstance(state, dict):
            # common keys: 'state_dict' or direct state-dict
            if "state_dict" in state:
                sd = state["state_dict"]
            else:
                sd = state
            # strip module. if present
            new_sd = {}
            for k,v in sd.items():
                nk = k[len("module."):] if k.startswith("module.") else k
                new_sd[nk] = v
            try:
                self.model.load_state_dict(new_sd)
                print(f"[INFO] Loaded state_dict from {ckpt}")
            except Exception as e:
                # fallback: maybe checkpoint is full model
                try:
                    self.model = state
                    print(f"[INFO] Loaded full model object from {ckpt}")
                except Exception as e2:
                    raise RuntimeError(f"Không thể load checkpoint: {e} | {e2}")
        else:
            # full model object pickled
            try:
                self.model = state
                print(f"[INFO] Loaded full model object from {ckpt}")
            except Exception as e:
                raise RuntimeError(f"Không thể load checkpoint: {e}")

        self.model = self.model.to(self.device).eval()

        # Transform: do a deterministic resize + to-tensor (we'll scale back using tensor size)
        # Albumentations Resize expects (height,width)
        self.tf = A.Compose([
            A.Resize(height=h, width=w),   # ensure deterministic shape
            ToTensorV2(),                  # returns tensor (dtype may be uint8 or float)
        ])

        # store resize dims for scaling
        self.input_h = h; self.input_w = w

    @staticmethod
    def _expand_and_clip(x1,y1,x2,y2,W,H,expand=0.0):
        if expand > 0:
            bw, bh = x2-x1, y2-y1
            cx, cy = x1 + bw/2, y1 + bh/2
            bw2, bh2 = bw*(1+expand), bh*(1+expand)
            x1, y1 = cx - bw2/2, cy - bh2/2
            x2, y2 = cx + bw2/2, cy + bh2/2
        x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
        x2 = min(W, int(round(x2))); y2 = min(H, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1,y1,x2,y2

    def process_image(self, image_path, out_dir="runs/rois", conf=0.30, topk=0, save_vis=False):
        """
        Run fasterrcnn on an image, crop ROIs and save them.
        Returns list of roi dicts (image, roi_path, score, label, coords)
        """
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Không thấy ảnh: {img_path}")

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Không đọc được ảnh: {img_path}")

        H, W = img.shape[:2]
        sample = self.tf(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bboxes=[], cls=[])
        tensor = sample["image"]  # ToTensorV2 result
        # ensure float in [0,1] because torchvision transform.normalize expects floats
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255.0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outs = self.model([tensor])[0]

        boxes = outs.get("boxes", torch.empty((0,4))).detach().cpu().numpy()
        scores = outs.get("scores", torch.empty((0,))).detach().cpu().numpy()
        labels = outs.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu().numpy()

        # filter background + conf
        keep = (labels != 0) & (scores >= conf)
        if keep.sum() == 0:
            return []

        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        order = np.argsort(-scores)
        if topk and topk > 0:
            order = order[:topk]
        boxes, scores, labels = boxes[order], scores[order], labels[order]

        # IMPORTANT: boxes are in coordinates relative to the model input tensor size
        # We resized original image to (input_w,input_h) before feeding; compute scale to original W,H
        in_h = tensor.shape[-2]; in_w = tensor.shape[-1]
        sx = float(W)/float(in_w); sy = float(H)/float(in_h)
        boxes[:, [0,2]] *= sx
        boxes[:, [1,3]] *= sy

        rois = []
        vis_img = img.copy() if (save_vis or self.vis_save) else None
        for i, (b,s,l) in enumerate(zip(boxes, scores, labels), 1):
            x1,y1,x2,y2 = b.tolist()
            rect = self._expand_and_clip(x1,y1,x2,y2,W,H,expand=self.expand)
            if rect is None: continue
            x1_,y1_,x2_,y2_ = rect
            crop = img[y1_:y2_, x1_:x2_]
            out_name = f"{img_path.stem}_roi{i}_{s:.3f}.jpg"
            out_file = out_dir / out_name
            cv2.imwrite(str(out_file), crop)
            rois.append({
                "image": str(img_path),
                "roi_path": str(out_file),
                "score": float(s),
                "label": int(l),
                "label_name": self.class_names[int(l)] if int(l) < len(self.class_names) else str(int(l)),
                "x1": int(x1_), "y1": int(y1_), "x2": int(x2_), "y2": int(y2_)
            })
            if vis_img is not None:
                cv2.rectangle(vis_img, (x1_,y1_), (x2_,y2_), (0,255,0), 2)
                txt = f"{rois[-1]['label_name']}:{s:.2f}"
                cv2.putText(vis_img, txt, (x1_, max(12,y1_-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        if vis_img is not None and (save_vis or self.vis_save):
            vis_path = out_dir / f"{img_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis_img)

        return rois

    def process_folder(self, folder, out_dir="runs/rois", conf=0.30, topk=0, save_vis=False):
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Không phải thư mục: {folder}")
        all_rois = []
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        for p in folder.rglob("*"):
            if p.suffix.lower() in exts:
                try:
                    r = self.process_image(p, out_dir=out_dir, conf=conf, topk=topk, save_vis=save_vis)
                    all_rois.extend(r)
                except Exception as e:
                    print("[WARN]", p.name, e)
        return all_rois

# Example usage:
if __name__ == "__main__":
    ext = FasterROIExtractor(
        weights=r"D:\PMT_Paper_Fasterrcnn-resnet50\output_newdataset\checkpoints\best_loss.pth",
        num_classes=2,
        resize_wh=(224,224),   # height,width used when creating input tensor
        class_names=("bg","palm"),
        vis_save=False
    )
    rois = ext.process_image(
        r"D:\PMT_\detect_roi\processed_dataset\autoUser1554\img_9.png",
        out_dir=r"D:\PMT_Paper_Fasterrcnn-resnet50\output_fasterrcnn_efficientnetb0_palm_newdataset\rois",
        conf=0.3,
        topk=1,
        save_vis=True
    )
    print(rois)
