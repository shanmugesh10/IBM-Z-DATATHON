
from pathlib import Path
import cv2
import numpy as np

RAW = Path("dataset/raw")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

images = sorted(
    p for p in RAW.glob("img_*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

masks = sorted(RAW.glob("ids_*.png"))

print("=== Synthetic Dataset Readiness ===")
print(f"RGB images found: {len(images)}")
print(f"ID masks found:   {len(masks)}")

if not images:
    print("\nWaiting for Role 2 to provide RGB images.")
    raise SystemExit(0)

ready = 0
missing_masks = 0
bad_pairs = 0

for image_path in images:
    mask_path = RAW / f"ids_{image_path.stem.removeprefix('img_')}.png"

    if not mask_path.exists():
        print(f"MISSING MASK: {image_path.name}")
        missing_masks += 1
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if image is None or mask is None:
        print(f"UNREADABLE PAIR: {image_path.name}")
        bad_pairs += 1
        continue

    if mask.ndim != 2:
        print(f"INVALID MASK CHANNELS: {mask_path.name}")
        bad_pairs += 1
        continue

    if mask.dtype != np.uint8:
        print(f"INVALID MASK TYPE: {mask_path.name}: {mask.dtype}")
        bad_pairs += 1
        continue

    if image.shape[:2] != mask.shape:
        print(f"SIZE MISMATCH: {image_path.name}")
        bad_pairs += 1
        continue

    values = np.unique(mask)
    unexpected = values[~np.isin(values, [0, 1])]

    if len(unexpected):
        print(f"UNEXPECTED IDS in {mask_path.name}: {unexpected[:20]}")
        bad_pairs += 1
        continue

    if not np.any(mask == 1):
        print(f"WARNING: No target pixels in {mask_path.name}")
        bad_pairs += 1
        continue

    ready += 1
    print(f"READY: {image_path.name} + {mask_path.name}")

print("\n=== Summary ===")
print(f"Ready pairs:   {ready}")
print(f"Missing masks: {missing_masks}")
print(f"Invalid pairs: {bad_pairs}")

if ready and not missing_masks and not bad_pairs:
    print("All discovered pairs passed these basic checks.")
else:
    print("Resolve the reported issues before generating labels.")
    