from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import io

app = FastAPI(
    title="IBM Datathon Object Detection API",
    description="Backend for object detection",
    version="1.0.0",
)

# Development/demo setting: permits the local HTML UI to call this API.
# For a deployed service, restrict allow_origins to your actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {"message": "IBM Datathon backend is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": False}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The file is not a valid image.")

    return {
        "message": "Image received successfully",
        "filename": Path(file.filename or "upload").name,
        "width": image.width,
        "height": image.height,
        "model_loaded": False,
        "prediction": None,
        "note": (
            "Image validation works. Object detection will be connected "
            "when the trained model is available."
        ),
    }
