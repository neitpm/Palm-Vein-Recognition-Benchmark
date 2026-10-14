# -*- coding: utf-8 -*-
"""
Module so khớp sinh trắc học tĩnh mạch (1:1 Verification và 1:N Identification).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class BiometricMatcher:
    """So khớp vector đặc trưng tĩnh mạch sử dụng Cosine Similarity hoặc Euclidean Distance."""

    def __init__(self, metric: str = "cosine", threshold: float = 0.65):
        """
        Args:
            metric: 'cosine' hoặc 'euclidean'. Mặc định 'cosine'.
            threshold: Ngưỡng quyết định chấp nhận/từ chối danh tính.
        """
        self.metric = metric.lower()
        self.threshold = threshold
        # Cơ sở dữ liệu đăng ký (Gallery / Enrolled Users): {user_id: [embedding_vector, ...]}
        self.gallery: Dict[str, List[np.ndarray]] = {}

    def enroll(self, user_id: str, embedding: np.ndarray):
        """Đăng ký mẫu sinh trắc tĩnh mạch mới vào hệ thống."""
        vec = embedding.flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        if user_id not in self.gallery:
            self.gallery[user_id] = []
        self.gallery[user_id].append(vec)

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Tính Cosine similarity giữa 2 vector embedding đã chuẩn hóa."""
        v1 = emb1.flatten()
        v2 = emb2.flatten()
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        v1 = v1 / n1
        v2 = v2 / n2
        if self.metric == "cosine":
            return float(np.dot(v1, v2))
        else:
            dist = np.linalg.norm(v1 - v2)
            return float(1.0 / (1.0 + dist))

    def verify(self, claimed_id: str, query_embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Xác thực 1:1 (Verification).
        Kiểm tra xem query_embedding có đúng là của claimed_id hay không.
        
        Returns:
            (is_match, max_similarity_score)
        """
        if claimed_id not in self.gallery or len(self.gallery[claimed_id]) == 0:
            return False, 0.0

        scores = [self.calculate_similarity(query_embedding, ref) for ref in self.gallery[claimed_id]]
        best_score = max(scores)
        is_match = best_score >= self.threshold
        return is_match, best_score

    def identify(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Nhận diện 1:N (Identification).
        Tìm xem query_embedding thuộc về ai trong toàn bộ Gallery.
        
        Returns:
            (best_user_id, highest_similarity_score)
        """
        if not self.gallery:
            return None, 0.0

        best_user = None
        highest_score = -1.0

        for user_id, templates in self.gallery.items():
            for ref in templates:
                score = self.calculate_similarity(query_embedding, ref)
                if score > highest_score:
                    highest_score = score
                    best_user = user_id

        if highest_score >= self.threshold:
            return best_user, highest_score
        return None, highest_score
