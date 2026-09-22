from ultralytics import YOLO
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import torch

model = YOLO("yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if device == "cuda" else -1
)

DEPTH_VERY_CLOSE = 180
DEPTH_NEARBY = 100


def get_position(bbox, frame_width):
    x_center = (bbox[0] + bbox[2]) / 2
    if x_center < frame_width * 0.33:
        return "left"
    elif x_center < frame_width * 0.66:
        return "ahead"
    else:
        return "right"


def get_proximity(bbox, frame_height):
    box_height = bbox[3] - bbox[1]
    ratio = box_height / frame_height
    if ratio > 0.6:
        return "very close"
    elif ratio > 0.3:
        return "nearby"
    else:
        return "in the distance"


def detect_objects(frame):
    results = model(frame, verbose=False)[0]
    detections = []
    frame_height, frame_width = frame.shape[:2]

    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < 0.5:
            continue
        bbox = box.xyxy[0].tolist()
        label = results.names[int(box.cls[0])]
        position = get_position(bbox, frame_width)
        proximity = get_proximity(bbox, frame_height)

        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "position": position,
            "proximity": proximity,
            "bbox": bbox
        })
    return detections, results


def build_narration(detections, lang='en'):
    if not detections:
        return "لا توجد عوائق. المسار يبدو خاليًا." if lang == 'ar' else "No obstacles detected. Path appears clear."

    left = [d['label'] for d in detections if d['position'] == 'left']
    ahead = [d['label'] for d in detections if d['position'] == 'ahead']
    right = [d['label'] for d in detections if d['position'] == 'right']
    danger = [d['label'] for d in detections if d['proximity'] == 'very close']

    if lang == 'ar':
        sentence = ""
        if danger:
            sentence += f"تحذير. {', '.join(danger)} قريب جدًا. "
        if ahead:
            sentence += f"أمامك مباشرة: {', '.join(ahead)}. "
        if left:
            sentence += f"على يسارك: {', '.join(left)}. "
        else:
            sentence += "الجانب الأيسر خالٍ. "
        if right:
            sentence += f"على يمينك: {', '.join(right)}. "
        else:
            sentence += "الجانب الأيمن خالٍ. "
        return sentence

    sentence = ""
    if danger:
        sentence += f"Warning. {', '.join(danger)} very close. "
    if ahead:
        sentence += f"Directly ahead: {', '.join(ahead)}. "
    if left:
        sentence += f"To your left: {', '.join(left)}. "
    else:
        sentence += "Left side clear. "
    if right:
        sentence += f"To your right: {', '.join(right)}. "
    else:
        sentence += "Right side clear. "
    return sentence


def estimate_depth(frame_bgr):
    frame_rgb = frame_bgr[:, :, ::-1]
    pil_img = Image.fromarray(frame_rgb)
    result = depth_estimator(pil_img)
    return np.array(result["depth"])


def get_distance_label(avg_depth):
    if avg_depth > DEPTH_VERY_CLOSE:
        return "very close"
    elif avg_depth > DEPTH_NEARBY:
        return "nearby, a few steps away"
    else:
        return "some distance away"


def get_ondemand_distance(frame, detections, lang='en'):
    if not detections:
        return ("لا يوجد شيء أمامك حاليًا" if lang == 'ar'
                else "There's nothing detected in front of you right now.")

    target = max(detections, key=lambda d: (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1]))

    try:
        depth_map = estimate_depth(frame)
    except Exception as e:
        print(f"DEPTH ERROR on-demand: {e}")
        label = target['proximity']
        return f"The {target['label']} is {label}."

    x1, y1, x2, y2 = [int(v) for v in target['bbox']]
    region = depth_map[y1:y2, x1:x2]
    avg_depth = float(np.median(region)) if region.size else 0
    distance_label = get_distance_label(avg_depth)

    if lang == 'ar':
        return f"{target['label']} {distance_label}"
    return f"The {target['label']} is {distance_label}."


last_detections = []
last_frame = None

def update_last_scan(detections, frame):
    global last_detections, last_frame
    last_detections = detections
    last_frame = frame


def answer_spatial_query(query_type, lang='en'):
    if not last_detections:
        return ("لا توجد بيانات حديثة" if lang == 'ar' else "I don't have a recent scan to check.")

    if query_type == 'clear':
        danger = [d for d in last_detections if d['proximity'] == 'very close']
        if danger:
            items = ', '.join(d['label'] for d in danger)
            return (f"لا، هناك {items} قريب جدًا" if lang == 'ar'
                    else f"No, there's {items} very close.")
        return "نعم، المسار يبدو خاليًا" if lang == 'ar' else "Yes, the path looks clear."

    side_objects = [d for d in last_detections if d['position'] == query_type]
    if not side_objects:
        side_word = {'left': 'يسارك' if lang=='ar' else 'your left',
                     'right': 'يمينك' if lang=='ar' else 'your right',
                     'ahead': 'أمامك' if lang=='ar' else 'ahead'}[query_type]
        return (f"لا شيء على {side_word}" if lang == 'ar'
                else f"Nothing detected on {side_word}.")

    items = ', '.join(f"{d['label']} ({d['proximity']})" for d in side_objects)
    return items