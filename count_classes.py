import os
from collections import defaultdict

CLASSES = [
    "person", "vehicle", "bicycle_motorcycle", "road", "sidewalk",
    "curb", "crosswalk", "traffic_light", "wall", "barrier",
    "tree_plant", "chair_furniture", "door", "stairs", "table", "laptop",
]

def count_class_presence(labels_dir):
    class_image_counts = defaultdict(int)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files. Counting...")

    for lf in label_files:
        label_path = os.path.join(labels_dir, lf)
        if os.path.getsize(label_path) == 0:
            continue
        seen_classes = set()
        with open(label_path) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    seen_classes.add(class_id)
        for c in seen_classes:
            class_image_counts[c] += 1

    print("\n--- Results ---")
    for class_id in sorted(class_image_counts.keys()):
        print(f"Class {class_id} ({CLASSES[class_id]}): {class_image_counts[class_id]} images")


count_class_presence(r"C:\Users\HP\Downloads\Extracted_File\training\yolo_labels")