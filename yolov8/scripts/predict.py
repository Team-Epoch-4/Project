import os
import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2
from config import TEST_IMG_DIR, PRED_SAVE_PATH, IDX2CAT_PATH

# 결과 저장 폴더 생성
PRED_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")

# 역매핑 로드
with open(IDX2CAT_PATH, "r") as f:
    idx2cat = {int(k): v for k, v in json.load(f).items()}

# 이미지 리스트
image_paths = sorted([p for p in TEST_IMG_DIR.glob("*.png")])

# 예측 저장
results_list = []
annotation_id = 1

for img_path in tqdm(image_paths, desc="예측 중"):
    results = model(str(img_path))[0]
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    image_id = img_path.stem

    for box in results.boxes:
        cls = int(box.cls.item())
        score = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # COCO 형식 bbox: [x, y, width, height]
        bbox_x = max(0, x1)
        bbox_y = max(0, y1)
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)

        results_list.append({
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": idx2cat[cls],
            "bbox_x": round(bbox_x, 2),
            "bbox_y": round(bbox_y, 2),
            "bbox_w": round(bbox_w, 2),
            "bbox_h": round(bbox_h, 2),
            "score": round(score, 4),
        })
        annotation_id += 1

# 저장
with open(PRED_SAVE_PATH / "predictions.json", "w") as f:
    json.dump(results_list, f, indent=2)

print("예측 완료 → predictions.json")