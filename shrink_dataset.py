import os
import random
from PIL import Image
import shutil

def resize_full_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir,
                          target_size=(640, 640)):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    valid_bases = []
    for lf in label_files:
        label_path = os.path.join(labels_dir, lf)
        if os.path.getsize(label_path) > 0:
            valid_bases.append(os.path.splitext(lf)[0])

    print(f"Processing all {len(valid_bases)} valid images (no sampling)...")

    for idx, base in enumerate(valid_bases):
        img_path = os.path.join(images_dir, base + '.jpg')
        label_path = os.path.join(labels_dir, base + '.txt')

        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as img:
            img_resized = img.convert('RGB').resize(target_size, Image.LANCZOS)
            img_resized.save(os.path.join(output_images_dir, base + '.jpg'), quality=85)

        shutil.copy(label_path, os.path.join(output_labels_dir, base + '.txt'))

        if idx % 1000 == 0:
            print(f"[{idx}/{len(valid_bases)}] processed")

    print(f"Done. {len(valid_bases)} images processed.")


resize_full_dataset(
    images_dir=r"C:\Users\HP\Downloads\Extracted_File\training\images",
    labels_dir=r"C:\Users\HP\Downloads\Extracted_File\training\yolo_labels",
    output_images_dir=r"C:\Users\HP\Desktop\vistas_shrunk\training\images",
    output_labels_dir=r"C:\Users\HP\Desktop\vistas_shrunk\training\yolo_labels",
)

resize_full_dataset(
    images_dir=r"C:\Users\HP\Downloads\Extracted_File\validation\images",
    labels_dir=r"C:\Users\HP\Downloads\Extracted_File\validation\yolo_labels",
    output_images_dir=r"C:\Users\HP\Desktop\vistas_shrunk\validation\images",
    output_labels_dir=r"C:\Users\HP\Desktop\vistas_shrunk\validation\yolo_labels",
)