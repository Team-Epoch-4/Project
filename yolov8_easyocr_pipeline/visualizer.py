import cv2
import numpy as np
from typing import List

COLOR_LIST = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

def draw_detections(image: np.ndarray, detections: List[dict], class_names: List[str]) -> np.ndarray:
    vis_img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls = det['class']
        color = COLOR_LIST[cls % len(COLOR_LIST)]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]}: {det['confidence']:.2f}"
        cv2.putText(vis_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis_img

def draw_ocr_results(image: np.ndarray, ocr_results: List[dict]) -> np.ndarray:
    vis_img = image.copy()
    for res in ocr_results:
        bbox = np.array(res['bbox']).astype(int)
        text = res['text']
        cv2.polylines(vis_img, [bbox], isClosed=True, color=(0,255,255), thickness=2)
        cv2.putText(vis_img, text, tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    return vis_img
