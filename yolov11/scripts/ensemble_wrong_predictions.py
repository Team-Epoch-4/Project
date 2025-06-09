import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou
import random

# ✅ 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
PILL_LIST_PATH = os.path.join(BASE_DIR, "configs", "pill_list.txt")
IMAGE_DIR = os.path.join(BASE_DIR, "yolo_dataset", "images", "val")
LABEL_DIR = os.path.join(BASE_DIR, "yolo_dataset", "labels", "val")
MODEL_M_PATH = os.path.join(BASE_DIR, "runs", "yolov11m", "exp", "weights", "best.pt")
MODEL_L_PATH = os.path.join(BASE_DIR, "runs", "yolov11l", "exp", "weights", "best.pt")
SAVE_DIR = os.path.join(BASE_DIR, "results", "ensemble_wrong_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 한글 폰트 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path) if os.path.exists(font_path) else None
mpl.rcParams['axes.unicode_minus'] = False

# ✅ 클래스 이름 로드
with open(PILL_LIST_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# ✅ 모델 로드
model_m = YOLO(MODEL_M_PATH)
model_l = YOLO(MODEL_L_PATH)

# ✅ 색상
random.seed(42)
colors = [(random.random(), random.random(), random.random()) for _ in class_names]

# ✅ 결과 집계
total_wrong = 0
image_names = sorted(os.listdir(IMAGE_DIR))

for img_name in image_names:
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

    result_m = model_m(img_path, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)[0]
    result_l = model_l(img_path, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)[0]

    if result_m.boxes is None or result_l.boxes is None:
        continue

    boxes_m = result_m.boxes.xyxy.cpu().numpy()
    confs_m = result_m.boxes.conf.cpu().numpy()
    mask = confs_m >= 0.5
    boxes_m = boxes_m[mask]
    confs_m = confs_m[mask]

    boxes_l = result_l.boxes.xyxy.cpu().numpy()
    classes_l = result_l.boxes.cls.cpu().numpy().astype(int)
    confs_l = result_l.boxes.conf.cpu().numpy()

    if len(boxes_m) == 0 or len(boxes_l) == 0:
        continue

    pred_labels = []
    final_classes = []
    final_confs = []

    for box_m in boxes_m:
        ious = np.array([bbox_iou(torch.tensor(box_m), torch.tensor(box_l)) for box_l in boxes_l])
        best_idx = np.argmax(ious)
        cls = classes_l[best_idx]
        conf = confs_l[best_idx]
        pred_labels.append(class_names[cls])
        final_classes.append(cls)
        final_confs.append(conf)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            gt_classes = [int(line.strip().split()[0]) for line in f.readlines()]
        gt_labels = [class_names[i] for i in sorted(gt_classes)]
    else:
        gt_labels = []

    if sorted(pred_labels) != gt_labels:
        total_wrong += 1
        print(f"\n🖼️ 이미지: {img_name}")
        print(f"📌 예측 클래스: {sorted(pred_labels)}")
        print(f"✅ 정답 클래스: {gt_labels}")

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        ax = plt.gca()

        for i, box in enumerate(boxes_m):
            cls = final_classes[i]
            conf = final_confs[i] * 100
            label = f"{class_names[cls]} {conf:.1f}%"
            x1, y1, x2, y2 = box
            color = colors[cls]

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label, color=color, fontsize=10,
                    fontproperties=font_prop,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.axis('off')
        plt.title("앙상블 틀린 예측", fontproperties=font_prop)
        save_path = os.path.join(SAVE_DIR, f"wrong_{img_name}")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"💾 저장됨: {save_path}")

print(f"\n❌ 앙상블 틀린 이미지 개수: {total_wrong}장")
