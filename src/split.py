
import random
import shutil
from pathlib import Path

ROOT = Path("dataset")
RAW = ROOT / "raw"
ALL_LABELS = ROOT / "labels_all"
CLASS_NAMES = ["part1"]
SEED = 42

images = sorted(RAW.glob("img_*.png"))

if not images:
    print("No synthetic RGB images found.")
    print("Waiting for Role 2's image-mask pairs.")
    raise SystemExit(0)

pairs = []

for image in images:
    label = ALL_LABELS / f"{image.stem}.txt"

    if label.exists() and label.read_text(encoding="utf-8").strip():
        pairs.append((image, label))
    else:
        print(f"Skipping missing or empty label: {image.name}")

if len(pairs) < 2:
    print("At least two labelled images are required for a split.")
    raise SystemExit(0)

random.Random(SEED).shuffle(pairs)

cut = int(0.8 * len(pairs))
cut = max(1, min(cut, len(pairs) - 1))

for folder in ("images/train", "images/val", "labels/train", "labels/val"):
    path = ROOT / folder
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

for split, items in (("train", pairs[:cut]), ("val", pairs[cut:])):
    for image, label in items:
        shutil.copy2(image, ROOT / "images" / split / image.name)
        shutil.copy2(label, ROOT / "labels" / split / label.name)

data_yaml = (
    f"path: {ROOT.resolve().as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: part1\n"
)

(ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")
Path("classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

print(f"Labelled images: {len(pairs)}")
print(f"Training images: {cut}")
print(f"Validation images: {len(pairs) - cut}")
print("Created dataset/data.yaml and classes.txt")
