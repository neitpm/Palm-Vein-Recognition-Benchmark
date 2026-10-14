# -*- coding: utf-8 -*-
"""
Module mạng nơ-ron nhận diện tĩnh mạch lòng bàn tay (Palm-Vein Deep Metric Learning).
Sử dụng backbone nhẹ (EfficientNet-B0 / MobileNetV3) kết hợp ArcMarginProduct (ArcFace).
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Head.
    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cosine similarity giữa normalized input và normalized weights
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))

        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class PalmVeinNet(nn.Module):
    """
    Mạng trích xuất đặc trưng sinh trắc tĩnh mạch lòng bàn tay (Feature Extractor).
    Đầu ra là vector embedding 512 chiều được L2-normalize.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: Optional[int] = None,
        weights: Optional[str] = "DEFAULT"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Backbone EfficientNet-B0 (nhẹ và độ chính xác cao)
        if weights == "DEFAULT":
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.backbone = efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        self.head = (
            ArcMarginProduct(embedding_dim, num_classes)
            if num_classes is not None and num_classes > 0
            else None
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất vector embedding L2-normalized 512-d từ ảnh x."""
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout(features)
        embedding = self.fc(features)
        embedding = self.bn2(embedding)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        embedding = self.extract_embedding(x)
        if self.head is not None:
            return self.head(embedding, label)
        return embedding
