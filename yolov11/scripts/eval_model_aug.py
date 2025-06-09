import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from ultralytics import YOLO
import random
from pathlib import Path

# ✅ 디렉토리 설정
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_PATH = BASE_DIR / "runs" / "yolov11l_aug" / "exp" / "weights" / "best.pt"
IMAGE_DIR = BASE_DIR / "yolo_dataset" / "images" / "val"
LABEL_DIR = BASE_DIR / "yolo_dataset" / "labels" / "val"
PILL_LIST_PATH = BASE_DIR / "configs" / "pill_list.txt"

# ✅ 한글 폰트 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path)
mpl.rcParams['axes.unicode_minus'] = False

# ✅ 클래스 이름 로드
with open(PILL_LIST_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# ✅ 시각화용 색상 설정
random.seed(42)
colors = [(random.random(), random.random(), random.random()) for _ in class_names]

# ✅ 모델 로드
print(f"\n🔍 YOLOv11-l_aug 예측 시작\n" + "-" * 50)
model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_DIR),
    conf=0.5,
    iou=0.5,
    agnostic_nms=True,
    save=False,
    verbose=False
)

total_wrong = 0

# ✅ 예측 결과 확인
for result in results:
    img_name = os.path.basename(result.path)
    label_path = LABEL_DIR / (Path(img_name).stem + ".txt")

    # 예측 클래스
    pred_classes = [int(c) for c in result.boxes.cls.cpu().numpy()]
    pred_labels = [class_names[i] for i in pred_classes]

    # 정답 클래스
    if label_path.exists():
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

        # 이미지 로드 및 시각화
        img_path = IMAGE_DIR / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        ax = plt.gca()

        for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
            cls = pred_classes[i]
            label = class_names[cls]
            conf = float(result.boxes.conf[i].cpu().numpy()) * 100
            label_text = f"{label} {conf:.1f}%"
            color = colors[cls]
            x1, y1, x2, y2 = box

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 edgecolor=color, facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label_text,
                    color=color, fontsize=10, fontproperties=font_prop,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.axis('off')
        plt.title("YOLOv11-l_aug 틀린 예측", fontproperties=font_prop)
        plt.show()

print(f"\n❌ YOLOv11-l_aug - 틀린 이미지 개수: {total_wrong}장")