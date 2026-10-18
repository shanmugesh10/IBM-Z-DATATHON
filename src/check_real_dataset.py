
from pathlib import Path

ROOT = Path("real_test")
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

images = sorted(
    p for p in IMAGE_DIR.glob("*")
    if p.is_file() and p.suffix.lower() in extensions
) if IMAGE_DIR.exists() else []

labels = sorted(LABEL_DIR.glob("*.txt")) if LABEL_DIR.exists() else []

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}

print("=== Real Evaluation Readiness ===")
print(f"Real images: {len(images)}")
print(f"Label files: {len(labels)}")
print(f"Images missing labels: {len(image_stems - label_stems)}")
print(f"Labels without images: {len(label_stems - image_stems)}")

if not images:
    print("Waiting for real photographs.")
elif image_stems - label_stems:
    print("Annotate the missing images before evaluation.")
elif label_stems - image_stems:
    print("Remove or correct orphan label files.")
else:
    print("Image-label filenames match. Validate the label contents next.")
