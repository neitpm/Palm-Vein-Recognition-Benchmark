# download_from_roboflow.py
"""
Unified dataset downloader pointing to shared `data/palm_dataset`.
"""
import os
import sys
from pathlib import Path

# Redirect to centralized downloader in project root
ROOT = Path(__file__).resolve().parents[2]
scripts_downloader = ROOT / "scripts" / "download_shared_dataset.py"

if scripts_downloader.is_file():
    exec(scripts_downloader.read_text(encoding="utf-8"))
else:
    from roboflow import Roboflow
    DATA_DIR = ROOT / "data" / "palm_dataset"
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(os.environ["ROBOFLOW_PROJECT"])
    ds = project.version(5).download("coco", location=str(DATA_DIR))
    print("Saved at:", ds.location)
