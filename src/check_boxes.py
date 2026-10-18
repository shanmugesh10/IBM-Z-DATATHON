
import cv2
import random
from pathlib import Path

RAW = Path("dataset/raw")
LABELS = Path("dataset/labels_all")
OUT = Path("dataset/check")
OUT.mkdir(parents=True, exist_ok=True)

images = sorted(RAW.glob("img_*.png"))

if not images:
    print("No RGB images found in dataset/raw/")
    raise SystemExit(0)

random.seed(42)
selected = random.sample(images, min(20, len(images)))

for image_path in selected:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path.name}")
        continue

    height, width = image.shape[:2]
    label_path = LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"Missing label: {label_path.name}")
        continue

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            print(f"Invalid label in {label_path.name}: {line}")
            continue

        class_id, x, y, w, h = values
        x, y, w, h = map(float, (x, y, w, h))

        x1 = round((x - w / 2) * width)
        y1 = round((y - h / 2) * height)
        x2 = round((x + w / 2) * width)
        y2 = round((y + h / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"class {class_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    output_path = OUT / f"check_{image_path.name}"
    cv2.imwrite(str(output_path), image)
    print(f"Saved: {output_path}")

print("\nDone. Inspect the images in dataset/check/")
