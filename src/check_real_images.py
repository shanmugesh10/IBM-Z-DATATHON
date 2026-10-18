
import cv2
from pathlib import Path

IMAGE_DIR = Path("real_test/images")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
) if IMAGE_DIR.exists() else []

if not images:
    print("No real photos found in real_test/images/")
    print("Ask Role 6 for the real photos or add your own.")
    raise SystemExit(0)

valid = 0
invalid = 0

for path in images:
    image = cv2.imread(str(path))

    if image is None:
        print(f"INVALID: {path.name}")
        invalid += 1
        continue

    height, width = image.shape[:2]
    print(f"OK: {path.name} | {width}x{height}")
    valid += 1

print("\nSummary")
print(f"Total files checked: {len(images)}")
print(f"Valid images: {valid}")
print(f"Unreadable images: {invalid}")
