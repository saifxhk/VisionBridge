import numpy as np
from PIL import Image
import os

CLASSES = [
    "person", "vehicle", "bicycle_motorcycle", "road", "sidewalk",
    "curb", "crosswalk", "traffic_light", "wall", "barrier",
    "tree_plant", "chair_furniture", "door", "stairs", "table", "laptop",
]

VISTAS_TO_OURS = {
    30: 0, 31: 0,
    107: 1, 108: 1, 109: 1, 113: 1, 114: 1, 115: 1,
    105: 2, 110: 2, 32: 2, 33: 2,
    21: 3,
    24: 4,
    4: 5,
    14: 6, 45: 6,
    90: 7, 91: 7, 92: 7, 93: 7, 94: 7, 95: 7,
    12: 8,
    5: 9, 6: 9, 8: 9, 9: 9, 10: 9, 11: 9,
    64: 10,
}


from scipy import ndimage

def instance_mask_to_yolo(instance_mask_path, output_txt_path, img_width, img_height,
                            min_area=100, max_box_fraction=0.5, grid_size=4):
    """
    Discrete objects (person, vehicle, etc.) still use normal connected-component
    boxes. Sprawling surface classes (road, barrier, tree-line, etc.) that would
    produce an oversized box get tiled into a grid instead — one box per grid
    cell that contains that class, rather than one giant box for the whole thing.
    """
    mask = np.array(Image.open(instance_mask_path))
    class_ids = mask // 256

    lines = []
    for vistas_id, our_class_id in VISTAS_TO_OURS.items():
        class_pixel_match = (class_ids == vistas_id)
        if not class_pixel_match.any():
            continue

        instance_values = np.unique(mask[class_pixel_match])

        for inst_val in instance_values:
            inst_mask = (mask == inst_val)
            ys, xs = np.where(inst_mask)
            if len(xs) < min_area:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box_w_frac = (x2 - x1) / img_width
            box_h_frac = (y2 - y1) / img_height

            if box_w_frac > max_box_fraction or box_h_frac > max_box_fraction:
                # Sprawling surface — tile into a fixed grid instead of one giant box
                tile_w = img_width // grid_size
                tile_h = img_height // grid_size

                for gy in range(grid_size):
                    for gx in range(grid_size):
                        tx1, tx2 = gx * tile_w, (gx + 1) * tile_w
                        ty1, ty2 = gy * tile_h, (gy + 1) * tile_h

                        tile_region = inst_mask[ty1:ty2, tx1:tx2]
                        if tile_region.sum() < min_area:
                            continue  # this tile has negligible/no presence of the class

                        tys, txs = np.where(tile_region)
                        # box tightly fit to the actual pixels within this tile,
                        # not the whole tile — keeps boxes accurate to real shape
                        bx1, bx2 = txs.min() + tx1, txs.max() + tx1
                        by1, by2 = tys.min() + ty1, tys.max() + ty1

                        x_center = ((bx1 + bx2) / 2) / img_width
                        y_center = ((by1 + by2) / 2) / img_height
                        width = (bx2 - bx1) / img_width
                        height = (by2 - by1) / img_height

                        lines.append(f"{our_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            else:
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                lines.append(f"{our_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(lines)

def process_all_vistas(instances_dir, images_dir, output_labels_dir, limit=None):
    os.makedirs(output_labels_dir, exist_ok=True)
    instance_files = [f for f in os.listdir(instances_dir) if f.endswith('.png')]

    if limit:
        instance_files = instance_files[:limit]

    total = len(instance_files)
    for idx, inst_file in enumerate(instance_files):
        base_name = os.path.splitext(inst_file)[0]
        inst_path = os.path.join(instances_dir, inst_file)

        img_path = os.path.join(images_dir, base_name + '.jpg')
        if not os.path.exists(img_path):
            print(f"Skipping {base_name} — no matching image found")
            continue
        with Image.open(img_path) as img:
            w, h = img.size

        output_path = os.path.join(output_labels_dir, base_name + '.txt')
        n_objects = instance_mask_to_yolo(inst_path, output_path, w, h)

        if idx % 500 == 0:
            print(f"[{idx}/{total}] {base_name}: {n_objects} objects")

    print(f"Done. Processed {len(instance_files)} images.")


# ── STEP 1: test on 5 images first, don't run full batch yet ──
# Training set — full batch
process_all_vistas(
    instances_dir=r"C:\Users\HP\Downloads\Extracted_File\training\v2.0\instances",
    images_dir=r"C:\Users\HP\Downloads\Extracted_File\training\images",
    output_labels_dir=r"C:\Users\HP\Downloads\Extracted_File\training\yolo_labels",
    limit=None
)

# Validation set — full batch
process_all_vistas(
    instances_dir=r"C:\Users\HP\Downloads\Extracted_File\validation\v2.0\instances",
    images_dir=r"C:\Users\HP\Downloads\Extracted_File\validation\images",
    output_labels_dir=r"C:\Users\HP\Downloads\Extracted_File\validation\yolo_labels",
    limit=None
)

