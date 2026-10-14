import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SSDROIExtractor:
    def __init__(self,
                 weights="output/checkpoints/best_loss.pth",
                 num_classes=2,
                 device=None,
                 resize_wh=(320,320),   # (width, height)
                 mean=(0.485,0.456,0.406),
                 std=(0.229,0.224,0.225),
                 class_names=("bg","palm"),
                 vis_save=False):

        self.project_root = ROOT
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.class_names = list(class_names)
        self.resize_wh = tuple(resize_wh)
        self.mean = mean; self.std = std
        self.vis_save = vis_save

        self.model = ssdlite320_mobilenet_v3_large(weights_backbone="DEFAULT", num_classes=num_classes)

        ckpt = Path(weights)
        if not ckpt.is_absolute():
            ckpt = self.project_root / ckpt
        if not ckpt.is_file():
            raise FileNotFoundError(f"Không thấy weights: {ckpt}")

        state = torch.load(str(ckpt), map_location=self.device)
        try:
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
            elif isinstance(state, dict) and all(k.startswith("module.") or k in self.model.state_dict() for k in state.keys()):
                sd = state
            else:
                sd = None
        except Exception:
            sd = None

        if sd is not None:
            new_sd = {}
            for k,v in sd.items():
                new_k = k
                if k.startswith("module."):
                    new_k = k[len("module."):]
                new_sd[new_k] = v
            try:
                self.model.load_state_dict(new_sd)
                print(f"[INFO] Loaded state_dict from {ckpt}")
            except Exception as e:
                try:
                    self.model = state
                    print(f"[INFO] Loaded full model object from {ckpt}")
                except Exception as e2:
                    raise RuntimeError(f"Không thể load checkpoint: {e} | {e2}")
        else:
            try:
                self.model = state
                print(f"[INFO] Loaded full model object from {ckpt}")
            except Exception as e:
                raise RuntimeError(f"Không thể load checkpoint: {e}")

        self.model = self.model.to(self.device).eval()

        w, h = self.resize_wh
        self.tf = A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(self.mean, self.std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls']))

    @staticmethod
    def _expand_and_clip(x1, y1, x2, y2, W, H, expand=0.0):
        if expand > 0:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw/2, y1 + bh/2
            bw2, bh2 = bw * (1 + expand), bh * (1 + expand)
            x1, y1 = cx - bw2/2, cy - bh2/2
            x2, y2 = cx + bw2/2, cy + bh2/2
        x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
        x2 = min(W, int(round(x2))); y2 = min(H, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _prepare_tensor(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample = self.tf(image=rgb, bboxes=[], cls=[])
        tensor = sample["image"]
        return tensor

    def process_image(self, image_path, out_dir="runs/rois", conf=0.30, expand=0.10, topk=0, save_vis=False):
        
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Không thấy ảnh: {img_path}")

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Không đọc được ảnh: {img_path}")
        H, W = img.shape[:2]

        tensor = self._prepare_tensor(img)
        in_h, in_w = tensor.shape[-2], tensor.shape[-1]

        with torch.no_grad():
            outs = self.model([tensor.to(self.device)])
            if isinstance(outs, (list, tuple)):
                out = outs[0]
            else:
                out = outs

            boxes = out.get("boxes", torch.empty((0,4))).detach().cpu().numpy()
            scores = out.get("scores", torch.empty((0,))).detach().cpu().numpy()
            labels = out.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu().numpy()

        keep = (labels != 0) & (scores >= conf)
        if keep.sum() == 0:
            return []  

        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        order = np.argsort(-scores)
        if topk and topk > 0:
            order = order[:topk]
        boxes, scores, labels = boxes[order], scores[order], labels[order]

        sx = float(W) / float(in_w)
        sy = float(H) / float(in_h)
        boxes[:, [0,2]] *= sx
        boxes[:, [1,3]] *= sy

        rois = []
        vis_img = img.copy() if (save_vis or self.vis_save) else None

        for i, (b, s, l) in enumerate(zip(boxes, scores, labels), 1):
            x1, y1, x2, y2 = b.tolist()
            rect = self._expand_and_clip(x1, y1, x2, y2, W, H, expand=expand)
            if rect is None:
                continue
            x1_, y1_, x2_, y2_ = rect
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
                cv2.rectangle(vis_img, (x1_, y1_), (x2_, y2_), (0,255,0), 2)
                txt = f"{rois[-1]['label_name']}:{s:.2f}"
                cv2.putText(vis_img, txt, (x1_, max(12, y1_ - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        if vis_img is not None and (save_vis or self.vis_save):
            vis_name = f"{img_path.stem}_vis.jpg"
            vis_path = out_dir / vis_name
            cv2.imwrite(str(vis_path), vis_img)

        return rois

    def process_folder(self, folder, out_dir="runs/rois", conf=0.30, expand=0.10, topk=0, save_vis=False):
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Không phải thư mục: {folder}")
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        all_rois = []
        for p in folder.rglob("*"):
            if p.suffix.lower() in exts:
                try:
                    rois = self.process_image(p, out_dir=out_dir, conf=conf, expand=expand, topk=topk, save_vis=save_vis)
                    all_rois.extend(rois)
                except Exception as e:
                    print(f"[WARN] {p.name}: {e}")
        return all_rois


if __name__ == "__main__":
    extractor = SSDROIExtractor(
        weights="output_newdataset/checkpoints/best_loss.pth",
        num_classes=2,
        resize_wh=(320,320),   # width, height
        class_names=("bg","palm"),
        vis_save=False
    )

    rois = extractor.process_image(
        image_path=r"D:\PMT_\detect_roi\processed_dataset\autoUser689\img_8.png",
        out_dir=r"D:\PMT_Paper\runs\rois",
        conf=0.30,
        expand=0.10,
        topk=1,
        save_vis=True
    )

    print("Extracted ROIs:", rois)
