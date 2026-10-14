# -*- coding: utf-8 -*-
"""
Module đánh giá độ chính xác sinh trắc học theo chuẩn quốc tế:
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- EER (Equal Error Rate)
- ROC Curve (Receiver Operating Characteristic)
"""

from typing import List, Tuple, Dict
import numpy as np


def compute_far_frr_eer(
    genuine_scores: List[float],
    impostor_scores: List[float],
    num_thresholds: int = 1000
) -> Dict[str, float]:
    """
    Tính EER, FAR và FRR từ danh sách điểm so khớp của cặp cùng người (genuine)
    và cặp khác người (impostor).
    """
    g_arr = np.array(genuine_scores)
    i_arr = np.array(impostor_scores)

    min_val = min(g_arr.min() if len(g_arr) else 0, i_arr.min() if len(i_arr) else 0)
    max_val = max(g_arr.max() if len(g_arr) else 1, i_arr.max() if len(i_arr) else 1)

    thresholds = np.linspace(min_val, max_val, num_thresholds)
    far_list = []
    frr_list = []

    for th in thresholds:
        # FAR: Impostor nhưng điểm >= threshold (nhận nhầm)
        far = np.mean(i_arr >= th) if len(i_arr) else 0.0
        # FRR: Genuine nhưng điểm < threshold (từ chối nhầm)
        frr = np.mean(g_arr < th) if len(g_arr) else 0.0

        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # EER là điểm mà |FAR - FRR| nhỏ nhất
    diff = np.abs(far_arr - frr_arr)
    idx_eer = np.argmin(diff)
    eer = (far_arr[idx_eer] + frr_arr[idx_eer]) / 2.0
    best_th = thresholds[idx_eer]

    return {
        "eer": float(eer),
        "optimal_threshold": float(best_th),
        "far_at_eer": float(far_arr[idx_eer]),
        "frr_at_eer": float(frr_arr[idx_eer])
    }
