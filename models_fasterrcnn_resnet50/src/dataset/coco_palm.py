# -*- coding: utf-8 -*-
import numpy as np, torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms_faster(train: bool):
    aug = []
    if train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.3, 0.3, 0.3, 0.3, p=0.5),
        ]
    aug += [ToTensorV2()]
    return A.Compose(
        aug,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["cls"])
    )
def get_transforms_resnet50_faster(train: bool):
    return get_transforms_faster(train)



class COCODataset(torch.utils.data.Dataset):
    """
    Y HỆT SSD: 1-class detector
    - background = 0, palm = 1
    """
    def __init__(self, img_root, ann_file, transform=None):
        self.coco_ds = CocoDetection(img_root, ann_file)
        self.ids = self.coco_ds.ids
        self.transform = transform

        used_cat_ids = sorted({a['category_id'] for a in self.coco_ds.coco.anns.values()})
        assert len(used_cat_ids) >= 1, "Không tìm thấy category_id trong annotations"

        self.cat2label = {used_cat_ids[0]: 1}
        self.label2cat = {1: used_cat_ids[0]}  # label 1 -> category_id gốc
        self.num_classes = 2                   # 0: background, 1: palm

    def __len__(self):
        return len(self.coco_ds)

    def __getitem__(self, idx):
        img, anns = self.coco_ds[idx]
        w, h = img.size

        boxes, labels = [], []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0: 
                continue
            x2, y2 = x + bw, y + bh
            if x2 > w or y2 > h or x < 0 or y < 0: 
                continue
            cat_id = a["category_id"]
            if cat_id not in self.cat2label: 
                continue
            boxes.append([x, y, x2, y2])
            labels.append(self.cat2label[cat_id])  # -> 1

        # Albumentations cần >=1 bbox nếu có bbox_params
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]; labels = [0]

        if self.transform:
            t = self.transform(image=np.array(img), bboxes=boxes, cls=labels)
            img = t["image"]; boxes, labels = t["bboxes"], t["cls"]

        # nếu thực sự không có GT
        if len(boxes) == 1 and labels[0] == 0:
            boxes, labels = [], []

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([self.ids[idx]]),
        }
        return img, target

def collate(batch):
    return tuple(zip(*batch))
