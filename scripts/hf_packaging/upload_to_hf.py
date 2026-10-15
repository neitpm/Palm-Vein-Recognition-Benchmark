#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload staged SSDLite Palm ROI model repository to Hugging Face Hub.
Usage:
    python scripts/hf_packaging/upload_to_hf.py --token <HF_TOKEN> [--repo-id neitpm/ssdlite-palm-vein-roi]
Or set environment variable:
    export HUGGING_FACE_HUB_TOKEN="your_token"
    python scripts/hf_packaging/upload_to_hf.py
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = REPO_ROOT / "hf_releases" / "ssdlite-palm-vein-roi"


def main():
    parser = argparse.ArgumentParser(description="Upload staged Palm ROI model repository to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="neitpm/ssdlite-palm-vein-roi", help="HF Repo ID (e.g. username/repo-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Create private repository if it does not exist")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not STAGING_DIR.is_dir() or not (STAGING_DIR / "README.md").is_file():
        print(f"[ERROR] Staging directory not found or empty: {STAGING_DIR}")
        print("[INFO] Please run `python scripts/hf_packaging/prepare_hf_model_repo.py` first.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] 'huggingface_hub' package not installed.")
        print("[INFO] Install via: pip install huggingface_hub")
        sys.exit(1)

    if not token:
        from huggingface_hub import HfFolder, login
        token = HfFolder.get_token()
        if not token:
            print("[INFO] No Hugging Face token detected.")
            print("[INFO] Please paste your Hugging Face write token (get it from https://huggingface.co/settings/tokens):")
            try:
                login()
                token = HfFolder.get_token()
            except Exception as e:
                print(f"[ERROR] Login failed: {e}")
                sys.exit(1)

    api = HfApi(token=token)

    print(f"[INFO] Creating/Verifying Hugging Face model repository: {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"[INFO] Repository ready: https://huggingface.co/{args.repo_id}")

    print(f"[INFO] Uploading staged files from {STAGING_DIR} to {args.repo_id}...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(STAGING_DIR),
        commit_message="Initial release of SSDLite-MobileNetV3 Palm ROI detector (PyTorch + ONNX FP32)",
    )

    print(f"[SUCCESS] Model uploaded successfully!")
    print(f"[INFO] View model card at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
