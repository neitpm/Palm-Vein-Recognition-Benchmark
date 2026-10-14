import sys
from pathlib import Path
import json
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models_fasterrcnn_resnet50.src.metrics.prf import detection_prf_cm as frcnn_prf
from models_ssdlite_mobilenetv3.src.metrics.prf import detection_prf_cm as ssd_prf
from models_fasterrcnn_resnet50.src.metrics.coco_eval import coco_map as frcnn_coco_map
from models_ssdlite_mobilenetv3.src.metrics.coco_eval import coco_map as ssd_coco_map


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


def test_coco_map_full_cocoeval():
    """Comprehensive test verifying actual COCOeval calculation, coordinate scaling,
    category mapping, multi-box predictions, and empty prediction handling."""
    
    # 1. Create a temporary synthetic COCO dataset with 3 images
    # Image 1 (id: 1): 200x200, GT box: [20, 20, 60, 60] (w=40, h=40), category_id: 1
    # Image 2 (id: 2): 400x200, GT box: [40, 20, 100, 60] (w=60, h=40), category_id: 1
    # Image 3 (id: 3): 100x100, GT box: [10, 10, 50, 50] (w=40, h=40), category_id: 1
    coco_dict = {
        "images": [
            {"id": 1, "width": 200, "height": 200},
            {"id": 2, "width": 400, "height": 200},
            {"id": 3, "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 20, 40, 40], "area": 1600, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [40, 20, 60, 40], "area": 2400, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "palm"}
        ]
    }
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(coco_dict, f)
        temp_json_path = f.name
        
    try:
        coco_gt = COCO(temp_json_path)
        
        # Batch 1: Image 1 & Image 2 (both input tensors resized to 100x100)
        # Note: orig_w, orig_h for Image 2 is 400x200, input is 100x100 -> sx = 4.0, sy = 2.0
        # In input space (100x100):
        # Image 1 (orig 200x200 -> sx=2, sy=2):
        #   GT [20, 20, 60, 60] in orig corresponds to [10, 10, 30, 30] in input space
        # Image 2 (orig 400x200 -> sx=4, sy=2):
        #   GT [40, 20, 100, 60] corresponds to [10, 10, 25, 30] in input space
        batch1_imgs = [torch.zeros(3, 100, 100), torch.zeros(3, 100, 100)]
        batch1_tgts = [{"image_id": torch.tensor([1])}, {"image_id": torch.tensor([2])}]
        
        # Batch 2: Image 3 (orig 100x100, input 100x100 -> sx=1, sy=1)
        batch2_imgs = [torch.zeros(3, 100, 100)]
        batch2_tgts = [{"image_id": torch.tensor([3])}]
        
        mock_loader = [
            (batch1_imgs, batch1_tgts),
            (batch2_imgs, batch2_tgts)
        ]
        
        # Predictions:
        # Pred 1 (Image 1): Box [10, 10, 30, 30] -> scaled to [20, 20, 60, 60] (Perfect match IoU=1.0)
        # Pred 2 (Image 2): Box [10, 10, 25, 30] -> scaled to [40, 20, 100, 60] (Perfect match IoU=1.0)
        #                   Plus a low score box [50, 50, 70, 70] (score 0.1 < conf_thres 0.3) -> should be filtered
        # Pred 3 (Image 3): No detection (empty box)
        preds = [
            # Image 1
            {
                "boxes": torch.tensor([[10., 10., 30., 30.]]),
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([1]) # label 1 -> maps to category 1
            },
            # Image 2
            {
                "boxes": torch.tensor([[10., 10., 25., 30.], [50., 50., 70., 70.]]),
                "scores": torch.tensor([0.90, 0.10]),
                "labels": torch.tensor([1, 1])
            },
            # Image 3
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64)
            }
        ]
        
        # Run Faster R-CNN coco_map
        model_frcnn = MockModel(preds)
        map_std, map_03 = frcnn_coco_map(
            model_frcnn, mock_loader, coco_gt, 
            device="cpu", conf_thres=0.30, label2cat={1: 1}
        )
        
        # Image 1: 1 GT, 1 TP (IoU 1.0) -> AP = 1.0
        # Image 2: 1 GT, 1 TP (IoU 1.0) -> AP = 1.0
        # Image 3: 1 GT, 0 TP -> AP = 0.0
        # Average Recall = 2 / 3 ≈ 0.667
        # Overall mAP@0.5:0.95 = 2 / 3 ≈ 0.6667
        # Overall mAP@0.3 = 2 / 3 ≈ 0.6667
        print(f"[COCOeval FRCNN Result] mAP@0.5:0.95 = {map_std:.4f}, mAP@0.3 = {map_03:.4f}")
        assert abs(map_std - (2.0 / 3.0)) < 1e-2, f"Expected mAP ~ 0.6667, got {map_std}"
        assert abs(map_03 - (2.0 / 3.0)) < 1e-2, f"Expected mAP@0.3 ~ 0.6667, got {map_03}"
        
        # Run SSDLite coco_map with identical loader
        model_ssd = MockModel(preds)
        map_std_ssd, map_03_ssd = ssd_coco_map(
            model_ssd, mock_loader, coco_gt, 
            device="cpu", conf_thres=0.30, label2cat={1: 1}
        )
        print(f"[COCOeval SSD Result] mAP@0.5:0.95 = {map_std_ssd:.4f}, mAP@0.3 = {map_03_ssd:.4f}")
        assert abs(map_std_ssd - (2.0 / 3.0)) < 1e-2, f"Expected SSD mAP ~ 0.6667, got {map_std_ssd}"
        assert abs(map_03_ssd - (2.0 / 3.0)) < 1e-2, f"Expected SSD mAP@0.3 ~ 0.6667, got {map_03_ssd}"
        
        print("[PASS] test_coco_map_full_cocoeval (Coordinate scaling, category mapping, multi-box filtering verified!)")
    finally:
        Path(temp_json_path).unlink(missing_ok=True)


def test_coco_map_edge_cases():
    """Verify edge cases:
    1. Image with NO Ground Truth annotations (background image) -> should penalize FP or ignore if no preds.
    2. Ground Truth with iscrowd=1 -> should be ignored in COCO evaluation as per standard protocol.
    3. Duplicate detections on the same ground truth box -> only the highest-score prediction matches TP, remaining are FP.
    4. Multiple categories -> ensures category filtering doesn't cross-contaminate.
    """
    coco_dict = {
        "images": [
            {"id": 10, "width": 100, "height": 100}, # Image with 1 palm GT
            {"id": 20, "width": 100, "height": 100}, # Image with iscrowd=1 GT
            {"id": 30, "width": 100, "height": 100}, # Pure background image (0 GT)
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 20, "category_id": 1, "bbox": [20, 20, 30, 30], "area": 900, "iscrowd": 1},
        ],
        "categories": [
            {"id": 1, "name": "palm"},
            {"id": 2, "name": "other"}
        ]
    }
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(coco_dict, f)
        temp_json_path = f.name
        
    try:
        coco_gt = COCO(temp_json_path)
        
        imgs = [torch.zeros(3, 100, 100), torch.zeros(3, 100, 100), torch.zeros(3, 100, 100)]
        tgts = [
            {"image_id": torch.tensor([10])},
            {"image_id": torch.tensor([20])},
            {"image_id": torch.tensor([30])}
        ]
        mock_loader = [(imgs, tgts)]
        
        # Predictions:
        # Image 10: 2 duplicate predictions on the single GT box [10, 10, 30, 30] (1 with score 0.95, 1 with score 0.85)
        #           Under COCO standard, the higher-score pred (0.95) matches GT (TP), second pred (0.85) is FP.
        # Image 20: 1 prediction on the iscrowd box (score 0.90)
        #           Under COCO protocol, detections matching iscrowd=1 GT are neither TP nor FP (ignored from matching count).
        # Image 30: 1 false positive prediction on background (score 0.80)
        #           No GT -> strictly FP.
        preds = [
            # Image 10
            {
                "boxes": torch.tensor([[10., 10., 40., 40.], [10., 10., 40., 40.]]),
                "scores": torch.tensor([0.95, 0.85]),
                "labels": torch.tensor([1, 1])
            },
            # Image 20
            {
                "boxes": torch.tensor([[20., 20., 50., 50.]]),
                "scores": torch.tensor([0.90]),
                "labels": torch.tensor([1])
            },
            # Image 30
            {
                "boxes": torch.tensor([[5., 5., 25., 25.]]),
                "scores": torch.tensor([0.80]),
                "labels": torch.tensor([1])
            }
        ]
        
        model = MockModel(preds)
        map_std, map_03 = frcnn_coco_map(
            model, mock_loader, coco_gt,
            device="cpu", conf_thres=0.30, label2cat={1: 1}
        )
        assert abs(map_std - 1.0) < 1e-4
        assert abs(map_03 - 1.0) < 1e-4

        # Direct white-box inspection of COCOeval internal matching arrays (evalImgs):
        # We instantiate COCOeval directly with the aggregated results to verify:
        # 1. dtMatches: detection 0 (score 0.95) matched GT id 1, while duplicate detection 1 (score 0.85) has match=0 (FP).
        # 2. dtIgnore: detection on iscrowd=1 GT (Image 20) has dtIgnore=True (ignored match, neither TP nor FP).
        # 3. gtIgnore: image 20 GT has gtIgnore=1, image 10 GT has gtIgnore=0.
        coco_dt = coco_gt.loadRes([
            {"image_id": 10, "category_id": 1, "bbox": [10.0, 10.0, 30.0, 30.0], "score": 0.95},
            {"image_id": 10, "category_id": 1, "bbox": [10.0, 10.0, 30.0, 30.0], "score": 0.85},
            {"image_id": 20, "category_id": 1, "bbox": [20.0, 20.0, 30.0, 30.0], "score": 0.90},
            {"image_id": 30, "category_id": 1, "bbox": [5.0, 5.0, 20.0, 20.0], "score": 0.80},
        ])
        direct_eval = COCOeval(coco_gt, coco_dt, "bbox")
        direct_eval.evaluate()
        
        # Find eval results filtered precisely by category_id, maxDet, and aRng:
        filtered_evals = [
            e for e in direct_eval.evalImgs 
            if e is not None 
            and e.get("category_id") == 1 
            and e.get("maxDet") == 100 
            and e.get("aRng") == [0, 1e5**2]
        ]
        
        # Verify that each image_id appears exactly once in the filtered results (no silent overwrites)
        img_ids_found = [e["image_id"] for e in filtered_evals]
        assert len(img_ids_found) == len(set(img_ids_found)), f"Duplicate evalImgs detected for same image_id: {img_ids_found}"
        assert set(img_ids_found) == {10, 20, 30}, f"Expected eval entries for {{10, 20, 30}}, got {set(img_ids_found)}"
        
        eval_by_img = {e["image_id"]: e for e in filtered_evals}
        
        # Verify Image 10 (duplicate detection handling):
        e10 = eval_by_img[10]
        # dtMatches shape is (10 IoU thresholds, 2 detections).
        # High-score pred (dt 0) must match GT 1 (dtMatches[:, 0] == 1.0)
        # Duplicate pred (dt 1) must be unmatched / FP (dtMatches[:, 1] == 0.0)
        assert np.all(e10["dtMatches"][:, 0] == 1.0), "Highest-score detection must match GT 1"
        assert np.all(e10["dtMatches"][:, 1] == 0.0), "Duplicate detection must NOT match GT 1 (treated as FP)"
        assert np.all(e10["dtIgnore"][:, 0] == False), "Top detection must NOT be ignored"
        assert np.all(e10["dtIgnore"][:, 1] == False), "Duplicate detection must NOT be ignored (penalized as FP)"
        
        # Verify Image 20 (iscrowd=1 handling):
        e20 = eval_by_img[20]
        assert np.all(e20["dtIgnore"] == True), "Detection matching iscrowd=1 GT must have dtIgnore=True"
        assert e20["gtIgnore"] == [1], "Ground truth with iscrowd=1 must have gtIgnore=1"
        
        # Verify Image 30 (pure background image):
        e30 = eval_by_img[30]
        assert np.all(e30["dtMatches"] == 0.0), "Detection on pure background must have dtMatches=0 (FP)"
        assert np.all(e30["dtIgnore"] == False), "Detection on background must not be ignored"
        assert len(e30["gtIgnore"]) == 0, "Background image has no GTs"
        
        print("[PASS] test_coco_map_edge_cases (Internal dtMatches, dtIgnore, gtIgnore, and duplicate FP verified!)")
    finally:
        Path(temp_json_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_frcnn_prf_multi_batch()
    test_ssd_prf_multi_batch()
    test_coco_map_full_cocoeval()
    test_coco_map_edge_cases()
    print("\nALL EVALUATION INTEGRITY TESTS PASSED!")
