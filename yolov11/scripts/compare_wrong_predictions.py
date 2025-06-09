import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ✅ 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))         # yolov11/scripts
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir)) # yolov11/
sys.path.append(BASE_DIR)  # configs 모듈 import 가능하게 설정

from configs.predict_config import YOLO_PREDICT_PARAMS

IMAGE_DIR = os.path.join(BASE_DIR, "yolo_dataset", "images", "val")
LABEL_DIR = os.path.join(BASE_DIR, "yolo_dataset", "labels", "val")
PILL_LIST_PATH = os.path.join(BASE_DIR, "configs", "pill_list.txt")

# ✅ 클래스 이름 불러오기
with open(PILL_LIST_PATH, encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

# ✅ IoU 계산 함수
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union != 0 else 0

# ✅ GT YOLO bbox 로드
def load_gt_boxes(label_path, img_w, img_h):
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return np.array(boxes), np.array(classes)
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return np.array(boxes), np.array(classes)

# ✅ 모델 경로들
model_paths = {
    "YOLOv11-S": os.path.join(BASE_DIR, "runs", "yolov11s", "exp", "weights", "best.pt"),
    "YOLOv11-M": os.path.join(BASE_DIR, "runs", "yolov11m", "exp", "weights", "best.pt"),
    "YOLOv11-L": os.path.join(BASE_DIR, "runs", "yolov11l", "exp", "weights", "best.pt"),
}

# ✅ 모델별 평가 루프
for model_name, model_path in model_paths.items():
    if not os.path.exists(model_path):
        print(f"\n📌 {model_name} 모델 파일 없음: {model_path}")
        continue

    print(f"\n🚀 {model_name} 예측 및 오류 분석 시작")
    model = YOLO(model_path)

    results = model.predict(
        source=IMAGE_DIR,
        conf=YOLO_PREDICT_PARAMS["conf"],
        iou=YOLO_PREDICT_PARAMS["iou"],
        agnostic_nms=YOLO_PREDICT_PARAMS["agnostic_nms"],
        save=False,
        verbose=False
    )

    class_error = 0
    bbox_error = 0

    for result in results:
        img_name = os.path.basename(result.path)
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        gt_boxes, gt_classes = load_gt_boxes(label_path, w, h)
        used_gt = set()

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy()

        for pb, pc in zip(pred_boxes, pred_classes):
            matched = False
            for i, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
                iou = compute_iou(pb, gb)
                if iou >= 0.5:
                    if int(pc) != int(gc):
                        class_error += 1  # 클래스 불일치
                    used_gt.add(i)
                    matched = True
                    break
            if not matched:
                bbox_error += 1  # false positive

        for i in range(len(gt_boxes)):
            if i not in used_gt:
                bbox_error += 1  # missed GT box

    # ✅ 결과 출력
    print(f"🌟 {model_name} 결과 요약")
    print(f"   - 분류 오류 개수 (클래스 불일치): {class_error}")
    print(f"   - BBox 오류 개수 (누락/과검출): {bbox_error}")