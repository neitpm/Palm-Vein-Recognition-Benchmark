# -*- coding: utf-8 -*-
"""
Centralized Dataset Downloader for Palm-Vein-Recognition-Benchmark.
Downloads a shared COCO dataset into `data/palm_dataset` for all models.
"""

import os
from pathlib import Path
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "palm_dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

api_key = os.environ.get("ROBOFLOW_API_KEY")
workspace = os.environ.get("ROBOFLOW_WORKSPACE")
project_id = os.environ.get("ROBOFLOW_PROJECT")

if not api_key or not workspace or not project_id:
    print("[ERROR] Please set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, and ROBOFLOW_PROJECT environment variables.")
    print("Example (PowerShell):")
    print('  $env:ROBOFLOW_API_KEY="your_key"')
    print('  $env:ROBOFLOW_WORKSPACE="your_workspace"')
    print('  $env:ROBOFLOW_PROJECT="your_project"')
    exit(1)

# Default to version 5 (or customize via ROBOFLOW_VERSION)
version_num = int(os.environ.get("ROBOFLOW_VERSION", 5))

print(f"[INFO] Initializing Roboflow download for project '{project_id}' version {version_num}...")
rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace).project(project_id)
ds = project.version(version_num).download("coco", location=str(DATA_DIR))

print(f"[SUCCESS] Dataset successfully saved to shared location: {ds.location}")
