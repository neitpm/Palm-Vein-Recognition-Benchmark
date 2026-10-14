# download_from_roboflow.py
import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(os.environ["ROBOFLOW_PROJECT"])
ds = project.version(1).download("coco", location=r"./roboflow_dl")
print("Saved at:", ds.location)
