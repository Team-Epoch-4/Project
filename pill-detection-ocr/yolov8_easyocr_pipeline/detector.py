from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple

class PillDetector:
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, image: np.ndarray) -> List[dict]:
        results = self.model(image, conf=self.conf_thres, iou=self.iou_thres)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls
                })
        return detections

    def crop_bboxes(self, image: np.ndarray, detections: List[dict]) -> List[Tuple[np.ndarray, dict]]:
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = image[y1:y2, x1:x2]
            crops.append((crop, det))
        return crops
