# -*- coding: utf-8 -*-
"""
End-to-End Palm-Vein Recognition Pipeline.
Quy trình:
1. Nhận ảnh thô / Stream từ camera.
2. SSD-MobileNetV3 phát hiện lòng bàn tay & crop ROI.
3. VeinEnhancer tiền xử lý làm nổi mạch máu (CLAHE + Gaussian blur).
4. PalmVeinNet trích xuất vector đặc trưng embedding (512 chiều).
5. BiometricMatcher đối soát xác thực (1:1 Verification) hoặc nhận diện (1:N Identification).
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import torch

from src.models.vein_enhancement import VeinEnhancer
from src.models.vein_recognizer import PalmVeinNet
from src.models.matcher import BiometricMatcher
from scripts.extract_rois_api import SSDROIExtractor


class PalmVeinRecognitionPipeline:
    def __init__(
        self,
        detector_weights: str = "output/checkpoints/best_loss.pth",
        recognizer_weights: Optional[str] = None,
        embedding_dim: int = 512,
        match_threshold: float = 0.65,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # 1. SSD ROI Detector
        self.detector = SSDROIExtractor(
            weights=detector_weights,
            num_classes=2,
            device=self.device,
            resize_wh=(320, 320),
            class_names=("bg", "palm"),
            vis_save=False
        )

        # 2. Vein Enhancer
        self.enhancer = VeinEnhancer(clip_limit=2.5, tile_grid_size=(8, 8), target_size=(224, 224))

        # 3. Recognition Feature Extractor
        self.recognizer = PalmVeinNet(embedding_dim=embedding_dim, num_classes=None, weights="DEFAULT" if recognizer_weights is None else None)
        if recognizer_weights and Path(recognizer_weights).is_file():
            state = torch.load(recognizer_weights, map_location=self.device)
            self.recognizer.load_state_dict(state, strict=False)
        self.recognizer = self.recognizer.to(self.device).eval()

        # 4. Biometric Matcher
        self.matcher = BiometricMatcher(metric="cosine", threshold=match_threshold)

    def extract_vein_embedding(self, bgr_roi: np.ndarray) -> np.ndarray:
        """Nhận diện tĩnh mạch từ ảnh ROI đã cắt."""
        enhanced_rgb = self.enhancer.enhance(bgr_roi, return_rgb=True)
        # Chuẩn hóa ImageNet
        norm_img = enhanced_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor_img = (norm_img - mean) / std
        tensor_img = torch.from_numpy(tensor_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            emb = self.recognizer.extract_embedding(tensor_img)
            return emb.squeeze(0).cpu().numpy()

    def process_and_enroll(self, user_id: str, image_path: str) -> Dict[str, Any]:
        """Phát hiện bàn tay, trích xuất đặc trưng và đăng ký User mới vào Database."""
        rois = self.detector.process_image(image_path, expand=0.10, topk=1)
        if not rois:
            return {"status": "error", "message": "No palm detected in image"}

        roi_path = rois[0]["roi_path"]
        roi_img = cv2.imread(roi_path)
        emb = self.extract_vein_embedding(roi_img)
        self.matcher.enroll(user_id, emb)
        return {
            "status": "success",
            "user_id": user_id,
            "score": rois[0]["score"],
            "box": [rois[0]["x1"], rois[0]["y1"], rois[0]["x2"], rois[0]["y2"]]
        }

    def verify_user(self, claimed_id: str, image_path: str) -> Dict[str, Any]:
        """Xác thực 1:1 danh tính."""
        rois = self.detector.process_image(image_path, expand=0.10, topk=1)
        if not rois:
            return {"status": "error", "message": "No palm detected in image"}

        roi_img = cv2.imread(rois[0]["roi_path"])
        query_emb = self.extract_vein_embedding(roi_img)
        is_match, score = self.matcher.verify(claimed_id, query_emb)
        return {
            "status": "success",
            "claimed_id": claimed_id,
            "is_match": bool(is_match),
            "similarity": float(score),
            "threshold": self.matcher.threshold
        }

    def identify_user(self, image_path: str) -> Dict[str, Any]:
        """Nhận diện 1:N danh tính từ toàn bộ Database."""
        rois = self.detector.process_image(image_path, expand=0.10, topk=1)
        if not rois:
            return {"status": "error", "message": "No palm detected in image"}

        roi_img = cv2.imread(rois[0]["roi_path"])
        query_emb = self.extract_vein_embedding(roi_img)
        user_id, score = self.matcher.identify(query_emb)
        return {
            "status": "success",
            "matched_user_id": user_id,
            "similarity": float(score),
            "threshold": self.matcher.threshold
        }
