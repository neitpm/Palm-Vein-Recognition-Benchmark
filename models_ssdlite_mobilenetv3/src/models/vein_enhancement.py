# -*- coding: utf-8 -*-
"""
Module tăng cường cấu trúc tĩnh mạch lòng bàn tay (Palm-Vein Enhancement).
Hỗ trợ CLAHE (Contrast Limited Adaptive Histogram Equalization) và lọc hình thái học/Gabor.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class VeinEnhancer:
    """Tăng cường độ tương phản mạch máu dưới da từ ảnh ROI bàn tay."""

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def enhance(
        self,
        bgr_image: np.ndarray,
        return_rgb: bool = True
    ) -> np.ndarray:
        """
        Nhận ảnh BGR ROI, chuẩn hóa kích thước, áp dụng CLAHE và làm mịn vi mô.
        
        Args:
            bgr_image: Ảnh BGR ROI cắt từ lòng bàn tay.
            return_rgb: Nếu True, chuyển ảnh kết quả về RGB 3 kênh để đưa vào backbone CNN.
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty or None")

        if len(bgr_image.shape) == 3:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_image

        # Resize chuẩn hóa kích thước phục vụ recognition backbone
        if gray.shape[:2] != self.target_size:
            gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

        # Cân bằng lược đồ màu thích ứng cục bộ (làm nổi mạch máu)
        enhanced = self.clahe.apply(gray)

        # Lọc nhiễu hạt nhẹ
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        if return_rgb:
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        return denoised
