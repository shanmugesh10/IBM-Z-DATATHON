
from ultralytics import YOLO
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Verify the GPU-enabled PyTorch installation first."
    )

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    seed=42,
    project="runs",
    name="synth_v1"
)
