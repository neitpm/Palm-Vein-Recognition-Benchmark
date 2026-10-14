# PMT_PAPER_FASTERRCNN-RESNET50 — Palm ROI Detection

Pipeline huấn luyện/đánh giá trích xuất ROI lòng bàn tay bằng **Faster R‑CNN**.
Repo hỗ trợ **2 biến thể**: (i) Faster R‑CNN ResNet50 (torchvision) và (ii) Faster R‑CNN với backbone **EfficientNet‑B0** (tùy biến). Có sẵn script đánh giá và script trích xuất ROI.

---

## 1) Cấu trúc thư mục

```
PMT_PAPER_FASTERRCNN-RESNET50/
├─ configs/                         # YAML cấu hình (đường dẫn dữ liệu, tham số train, …)
├─ output/                          # Kết quả/chạy mặc định
├─ output_fasterrcnn_efficientnetb0_palm/        # Kết quả huấn luyện EfficientNet-B0
│  ├─ checkpoints/                  # *.pt / *.pth
│  ├─ diagnostics/                  # Thông tin chẩn đoán (nếu lưu)
│  ├─ logs/                         # train_log.txt, tensorboard, v.v.
│  ├─ plots/
│  │  ├─ loss_curve.png
│  │  └─ metrics_curve.png
│  ├─ rois/                         # Ảnh đã trích xuất ROI (tùy phiên bản)
│  ├─ metrics_epoch.json            # Chỉ số theo epoch
│  └─ metrics_summary.json          # Tổng kết chỉ số (mAP, P/R/F1...)
├─ output_newdataset/               # Kết quả khi dùng bộ dữ liệu khác
│  ├─ checkpoints/ diagnostics/ logs/ plots/
├─ scripts/
│  ├─ eval_fasterrcnn_resnet50_palm.py           # Đánh giá mô hình
│  ├─ fasterrcnn_roi_extractor.py                # Script/API trích xuất ROI
│  ├─ train_fasterrcnn_efficientnetb0_palm.py    # Huấn luyện bản EfficientNet‑B0
│  └─ train_fasterrcnn_resnet50_palm.py          # Huấn luyện bản ResNet50
├─ src/                              # Mã nguồn lõi (dataset, model utils, metrics, ...)
├─ .env                              # (khuyến nghị) Biến môi trường cục bộ
├─ .gitattributes
├─ .gitignore
├─ README.md
└─ requirements.txt
```

---

## 2) Yêu cầu hệ thống

- Python 3.9+ (khuyến nghị 3.10/3.11)
- PyTorch + TorchVision phù hợp bản CUDA/Driver (nếu dùng GPU)
- Các gói khác trong `requirements.txt`
---

## 3) Cài đặt nhanh

```bash
# 1) Tạo & kích hoạt môi trường ảo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2) Cài PyTorch (chọn đúng bản từ pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # ví dụ CUDA 12.1

# 3) Cài phụ thuộc còn lại
pip install -r requirements.txt
```

---

## 4) Dữ liệu

### 4.1 Định dạng
- Hỗ trợ **COCO**: thư mục `images/` và file nhãn `annotations/{train,val,test}.json`.
- Nếu dùng cấu trúc khác, chỉnh lại lớp `Dataset` trong `src/` hoặc adapter tương ứng.

### 4.2 Khai báo đường dẫn
- Cập nhật các file **YAML** trong `configs/` (ví dụ `data.train`, `data.val`, `data.test`, `data.num_classes`, v.v.),
  hoặc truyền tham số qua CLI nếu script hỗ trợ.

---

## 5) Huấn luyện

### 5.1 ResNet50
```bash
# Dùng cấu hình trong YAML (khuyến nghị)
python scripts/train_fasterrcnn_resnet50_palm.py --config configs/fasterrcnn_resnet50.yaml
# (Nếu script không có --config, xem phần argparse trong file và truyền đối số tương ứng)
```

### 5.2 EfficientNet‑B0
```bash
python scripts/train_fasterrcnn_efficientnetb0_palm.py --config configs/fasterrcnn_efficientnetb0.yaml
```

**Kết quả** sẽ nằm trong `output*/` gồm:
- `checkpoints/`: mô hình đã lưu (*.pt/*.pth)
- `logs/` và `train_log.txt`: log theo thời gian
- `plots/`: `loss_curve.png`, `metrics_curve.png`
- `metrics_epoch.json`, `metrics_summary.json`: chỉ số mAP/P/R/F1 theo epoch và tổng kết

> Gợi ý: cấu hình `seed`, `batch_size`, `lr`, `epochs`, `img_size/transform` trong YAML để tái lập kết quả.

---

## 6) Đánh giá

```bash
python scripts/eval_fasterrcnn_resnet50_palm.py \
  --config configs/fasterrcnn_resnet50.yaml \
  --ckpt path/to/checkpoint.pth
```

- Tính **mAP** theo COCO và các chỉ số P/R/F1 (tùy implementation).
- Lưu biểu đồ/JSON vào `plots/` và `metrics_*` trong thư mục output tương ứng.

---

## 7) Trích xuất ROI

Có 2 cách phổ biến:

### 7.1 Batch/CLI
```bash
python scripts/fasterrcnn_roi_extractor.py \
  --ckpt path/to/checkpoint.pth \
  --input_dir path/to/images \
  --save_dir output_fasterrcnn_efficientnetb0_palm/rois
```

> Kết quả: toạ độ bbox và/hoặc ảnh đã cắt ROI, lưu về `rois/`.

---

## 8) Nhật ký & Chẩn đoán

- `logs/` + `train_log.txt`: tiến trình huấn luyện (loss, lr, thời gian/epoch…).
- `diagnostics/`: thông tin kiểm tra dữ liệu/mô hình (nếu bật).
- `metrics_epoch.json`: danh sách chỉ số theo epoch.
- `metrics_summary.json`: tổng hợp chỉ số tốt nhất (best mAP, best F1…).

Bạn có thể vẽ biểu đồ tuỳ chỉnh từ JSON hoặc xem nhanh trong `plots/`.

---

## 9) Troubleshooting

- **Đường dẫn có khoảng trắng** → luôn đặt trong ngoặc kép.
- **CUDA**: kiểm tra `torch.version.cuda` và driver; cài đúng bản `torch/torchvision`.
- **OOM (hết VRAM)**: giảm `batch_size`, tăng `stride`/`min_size`, tắt augment nặng.
- **mAP = 0**: kiểm tra mapping `category_id`, lớp background, và khớp tên/ID trong COCO.
- **Không load được checkpoint**: khớp kiến trúc (ResNet50 vs EfficientNet‑B0), phiên bản PyTorch.

---

## 10) Quy trình khuyến nghị

1. Chuẩn bị dữ liệu COCO, kiểm tra `category_id` và kích thước ảnh.
2. Cập nhật `configs/` (đường dẫn + siêu tham số + num_classes).
3. Huấn luyện (ResNet50 hoặc EfficientNet‑B0).
4. Đánh giá → xem `metrics_*` và `plots/`.
5. Trích xuất ROI cho downstream task (classification, matching, v.v.).

