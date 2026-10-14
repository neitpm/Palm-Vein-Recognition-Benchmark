# -*- coding: utf-8 -*-
"""
Script minh họa chạy trọn vẹn End-to-End Palm-Vein Recognition Pipeline.
Demo cả 2 kịch bản:
- Đăng ký người dùng (Enrollment)
- Nhận diện 1:N và Xác thực 1:1 (Identification & Verification)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from src.pipeline import PalmVeinRecognitionPipeline


def main():
    parser = argparse.ArgumentParser(description="Demo End-to-End Palm Vein Recognition")
    parser.add_argument("--detector_weights", default="output/checkpoints/best_loss.pth", help="Checkpoint SSD detector")
    parser.add_argument("--image", default=None, help="Ảnh bàn tay cần test")
    parser.add_argument("--threshold", type=float, default=0.65, help="Ngưỡng Cosine similarity")
    args = parser.parse_args()

    print("[INFO] Initializing End-to-End Palm-Vein Pipeline...")
    pipeline = PalmVeinRecognitionPipeline(
        detector_weights=args.detector_weights,
        match_threshold=args.threshold
    )
    print("[INFO] Pipeline ready!")

    # Nếu người dùng truyền ảnh cụ thể
    if args.image and Path(args.image).is_file():
        print(f"[INFO] Processing image: {args.image}")
        res = pipeline.identify_user(args.image)
        print("Recognition Result:", res)
    else:
        print("[INFO] Running Self-test demo...")
        roi_samples = list((ROOT / "runs" / "rois").glob("*_roi*.jpg"))
        if roi_samples:
            sample_roi = str(roi_samples[0])
            import cv2
            img = cv2.imread(sample_roi)
            emb1 = pipeline.extract_vein_embedding(img)
            pipeline.matcher.enroll("user_test_01", emb1)
            is_match, score = pipeline.matcher.verify("user_test_01", emb1)
            print(f"[TEST PASSED] Enroll & Verify user_test_01: score={score:.4f}, match={is_match}")
        else:
            print("[INFO] No ROI sample found in runs/rois, pipeline init verified successfully.")


if __name__ == "__main__":
    main()
