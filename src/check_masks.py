
import cv2
import numpy as np
from pathlib import Path

RAW = Path("dataset/raw")

mask_files = sorted(RAW.glob("ids_*.png"))

if not mask_files:
    print("No ID masks found in dataset/raw/")
    print("Waiting for Role 2 to provide ids_XXXX.png files.")
    raise SystemExit(0)

print("Checking ID masks...\n")

for mask_path in mask_files[:10]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        print(f"{mask_path.name}: ERROR - unreadable file")
        continue

    print(f"File: {mask_path.name}")
    print(f"Shape: {mask.shape}")
    print(f"Data type: {mask.dtype}")
    print(f"Unique pixel values: {np.unique(mask)[:30]}")

    if mask.ndim != 2:
        print("WARNING: Mask is not single-channel grayscale.")

    if mask.dtype != np.uint8:
        print("WARNING: Mask is not 8-bit.")

    print("-" * 40)

print("\nMask check complete.")
