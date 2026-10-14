import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock

from models_fasterrcnn_resnet50.src.metrics.prf import detection_prf_cm as frcnn_prf
from models_ssdlite_mobilenetv3.src.metrics.prf import detection_prf_cm as ssd_prf
from models_fasterrcnn_resnet50.src.metrics.coco_eval import coco_map as frcnn_coco_map


class MockModel(torch.nn.Module):
    def __init__(self, pred_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.idx = 0

    def forward(self, imgs):
        batch_size = len(imgs)
        batch_res = self.pred_outputs[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return batch_res


def test_frcnn_prf_multi_batch():
    # 2 batches, each with 2 images = 4 images total
    # Image 0: Perfect match (1 pred == 1 GT)
    # Image 1: False positive (1 pred, 0 GT)
    # Image 2: False negative (0 pred, 1 GT)
    # Image 3: 1 match, 1 extra pred (1 TP, 1 FP)
    
    batch1_imgs = [torch.zeros(3, 100, 100), torch.zeros(3, 100, 100)]
    batch1_tgts = [
        {"boxes": torch.tensor([[10., 10., 50., 50.]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)},
    ]
    
    batch2_imgs = [torch.zeros(3, 100, 100), torch.zeros(3, 100, 100)]
    batch2_tgts = [
        {"boxes": torch.tensor([[20., 20., 60., 60.]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[10., 10., 40., 40.]]), "labels": torch.tensor([1])},
    ]
    
    preds = [
        # Img 0
        {"boxes": torch.tensor([[10., 10., 50., 50.]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])},
        # Img 1
        {"boxes": torch.tensor([[10., 10., 50., 50.]]), "scores": torch.tensor([0.8]), "labels": torch.tensor([1])},
        # Img 2
        {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)},
        # Img 3
        {"boxes": torch.tensor([[10., 10., 40., 40.], [50., 50., 80., 80.]]), "scores": torch.tensor([0.95, 0.7]), "labels": torch.tensor([1, 1])},
    ]
    
    mock_loader = [
        (batch1_imgs, batch1_tgts),
        (batch2_imgs, batch2_tgts)
    ]
    
    model = MockModel(preds)
    p, r, f1, cm, per_class = frcnn_prf(model, mock_loader, num_classes=2, iou_thr=0.5, score_thr=0.5, device="cpu")
    
    # Total GTs: 1 (img0) + 0 (img1) + 1 (img2) + 1 (img3) = 3 GTs
    # Total Preds: 1 (img0) + 1 (img1) + 0 (img2) + 2 (img3) = 4 Preds
    # Matches (TP): img0 (1 TP), img1 (0 TP), img2 (0 TP), img3 (1 TP) => Total TP = 2
    # FP = Total Preds - TP = 4 - 2 = 2
    # FN = Total GTs - TP = 3 - 2 = 1
    # Expected Precision = TP / (TP + FP) = 2 / 4 = 0.5
    # Expected Recall = TP / (TP + FN) = 2 / 3 ≈ 0.6667
    
    assert abs(p - 0.5) < 1e-4, f"Expected P=0.5, got {p}"
    assert abs(r - (2.0 / 3.0)) < 1e-4, f"Expected R={2.0/3.0}, got {r}"
    assert abs(f1 - (2 * 0.5 * (2/3)) / (0.5 + 2/3)) < 1e-4
    print("[PASS] test_frcnn_prf_multi_batch")


def test_ssd_prf_multi_batch():
    batch1_imgs = [torch.zeros(3, 320, 320), torch.zeros(3, 320, 320)]
    batch1_tgts = [
        {"boxes": torch.tensor([[10., 10., 50., 50.]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)},
    ]
    
    batch2_imgs = [torch.zeros(3, 320, 320), torch.zeros(3, 320, 320)]
    batch2_tgts = [
        {"boxes": torch.tensor([[20., 20., 60., 60.]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[10., 10., 40., 40.]]), "labels": torch.tensor([1])},
    ]
    
    preds = [
        {"boxes": torch.tensor([[10., 10., 50., 50.]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[10., 10., 50., 50.]]), "scores": torch.tensor([0.8]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)},
        {"boxes": torch.tensor([[10., 10., 40., 40.], [50., 50., 80., 80.]]), "scores": torch.tensor([0.95, 0.7]), "labels": torch.tensor([1, 1])},
    ]
    
    mock_loader = [
        (batch1_imgs, batch1_tgts),
        (batch2_imgs, batch2_tgts)
    ]
    
    model = MockModel(preds)
    p, r, f1, cm, per_class = ssd_prf(model, mock_loader, num_classes=2, iou_thr=0.5, score_thr=0.5, device="cpu")
    
    assert abs(p - 0.5) < 1e-4
    assert abs(r - (2.0 / 3.0)) < 1e-4
    print("[PASS] test_ssd_prf_multi_batch")


def test_frcnn_coco_map_all_batches():
    # Verify that coco_map actually aggregates results across ALL batches rather than just the last batch
    coco_gt = MagicMock()
    coco_gt.imgs = {
        101: {'width': 100, 'height': 100},
        102: {'width': 100, 'height': 100},
    }
    
    # 2 batches with 1 image each
    batch1_imgs = [torch.zeros(3, 100, 100)]
    batch1_tgts = [{"image_id": torch.tensor([101])}]
    
    batch2_imgs = [torch.zeros(3, 100, 100)]
    batch2_tgts = [{"image_id": torch.tensor([102])}]
    
    preds = [
        {"boxes": torch.tensor([[10., 10., 50., 50.]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[20., 20., 60., 60.]]), "scores": torch.tensor([0.85]), "labels": torch.tensor([1])},
    ]
    
    mock_loader = [
        (batch1_imgs, batch1_tgts),
        (batch2_imgs, batch2_tgts)
    ]
    
    model = MockModel(preds)
    
    # Intercept loadRes to check how many detections were aggregated
    recorded_results = []
    def mock_loadRes(res):
        recorded_results.extend(res)
        mock_dt = MagicMock()
        return mock_dt
        
    coco_gt.loadRes = mock_loadRes
    
    # Run coco_map
    map_std, map_03 = frcnn_coco_map(model, mock_loader, coco_gt, device="cpu", conf_thres=0.5, label2cat={1: 1})
    
    # Must have aggregated predictions for both image 101 AND 102
    assert len(recorded_results) == 2, f"Expected 2 aggregated results, got {len(recorded_results)}"
    image_ids = {r["image_id"] for r in recorded_results}
    assert image_ids == {101, 102}, f"Expected image_ids {{101, 102}}, got {image_ids}"
    print("[PASS] test_frcnn_coco_map_all_batches")


if __name__ == "__main__":
    test_frcnn_prf_multi_batch()
    test_ssd_prf_multi_batch()
    test_frcnn_coco_map_all_batches()
    print("\nALL EVALUATION INTEGRITY TESTS PASSED!")
