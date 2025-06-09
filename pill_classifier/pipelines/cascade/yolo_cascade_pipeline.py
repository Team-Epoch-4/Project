
from ultralytics import YOLO
import cv2
import os
import numpy as np

def load_yolo_model(weight_path):
    """
    YOLOv8 모델 weight를 로드합니다.
    
    Args:
        weight_path (str): .pt 파일 경로
    Returns:
        YOLO: 로드된 YOLO 모델
    """
    return YOLO(weight_path)

def get_cropped_bboxes(image_path, yolo_model, conf_thresh=0.5, return_xyxy=False):
    """
    YOLO 모델을 이용해 이미지 내 객체를 탐지하고, 모든 bbox를 crop하여 반환합니다.
    
    Args:
        image_path (str): 대상 이미지 경로
        yolo_model (YOLO): 로드된 YOLO 모델 객체
        conf_thresh (float): confidence threshold
        return_xyxy (bool): True일 경우 (crop, bbox좌표) 반환

    Returns:
        list: [(crop, (x1, y1, x2, y2))] or [crop]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")

    results = yolo_model.predict(source=image_path, conf=conf_thresh, verbose=False)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"❌ cv2로 이미지를 불러오지 못했습니다: {image_path}")

    crops = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if return_xyxy:
            crops.append((crop, (x1, y1, x2, y2)))
        else:
            crops.append(crop)

    return crops
