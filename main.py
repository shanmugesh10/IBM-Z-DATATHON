
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import io

app = FastAPI(
    title="IBM Datathon Object Detection API",
    description="YOLOv8 object detection backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent / "yolov8n.pt"
model = YOLO(str(MODEL_PATH))

@app.get("/")
def home():
    return {"message": "IBM Datathon backend is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty."
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail="Image must be 10 MB or smaller."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400, detail="Invalid image file."
        )

    results = model.predict(image, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [
                round(float(v), 2)
                for v in box.xyxy[0].tolist()
            ],
        })

    return {
        "filename": Path(file.filename or "upload").name,
        "width": image.width,
        "height": image.height,
        "model_loaded": True,
        "detections": detections,
        "count": len(detections),
    }
