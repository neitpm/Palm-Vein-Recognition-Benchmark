# Dataset Structure (COCO Format)

The dataset layout adheres to standard COCO object detection format:

```
dataset_root/
├─ train/
│  ├─ _annotations.coco.json
│  └─ images...
├─ valid/
│  ├─ _annotations.coco.json
│  └─ images...
```

## Downloading via Roboflow
Set your API key and download:
```bash
python download_data.py
```
