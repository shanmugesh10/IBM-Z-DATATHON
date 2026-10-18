
import cv2
import numpy as np
from pathlib import Path

RAW = Path("dataset/raw")
OUT = Path("dataset/labels_all")
OUT.mkdir(parents=True, exist_ok=True)

ID_TO_CLASS = {1: 0}
MIN_PIXELS = 100

image_files = sorted(RAW.glob("img_*.png"))

if not image_files:
    print("No RGB images found in dataset/raw/")
    raise SystemExit(0)

total = 0
empty = 0
boxes = 0

for image_path in image_files:
    mask_path = RAW / image_path.name.replace("img_", "ids_", 1)

    if not mask_path.exists():
        print(f"Missing mask for {image_path.name}")
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if image is None or mask is None:
        print(f"Unreadable image or mask: {image_path.name}")
        continue

    if mask.ndim == 3:
        print(f"ERROR: Mask must be grayscale: {mask_path.name}")
        continue

    height, width = image.shape[:2]

    if mask.shape != (height, width):
        print(f"ERROR: Size mismatch: {image_path.name}")
        continue

    lines = []

    for object_id, class_id in ID_TO_CLASS.items():
        ys, xs = np.where(mask == object_id)

        if len(xs) < MIN_PIXELS:
            continue

        x_min = xs.min()
        x_max = xs.max() + 1
        y_min = ys.min()
        y_max = ys.max() + 1

        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    label_path = OUT / f"{image_path.stem}.txt"
    label_path.write_text("\n".join(lines), encoding="utf-8")

    total += 1
    boxes += len(lines)

    if not lines:
        empty += 1

print(f"Images processed: {total}")
print(f"Bounding boxes generated: {boxes}")
print(f"Images without boxes: {empty}")
print(f"Labels saved to: {OUT.resolve()}")
