import cv2
from utils.detection import estimate_depth

import numpy as np

distances_to_test = [1, 2, 3, 6, 9]  # meters
results = {}

cap = cv2.VideoCapture(0)

for dist in distances_to_test:
    input(f"\nPlace an object exactly {dist}m from the camera, centered in frame. Press Enter when ready...")
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed, skipping")
        continue

    depth_map = estimate_depth(frame)
    h, w = depth_map.shape
    center_region = depth_map[h//2-20:h//2+20, w//2-20:w//2+20]
    avg_value = float(np.mean(center_region))
    results[dist] = avg_value
    print(f"  -> depth value at {dist}m: {avg_value:.2f}")

cap.release()

print("\n--- Calibration Results ---")
for dist, val in results.items():
    print(f"{dist}m -> {val:.2f}")