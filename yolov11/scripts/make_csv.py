import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# ✅ 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent          # yolov11/scripts
BASE_DIR = SCRIPT_DIR.parent                          # yolov11/
MODEL_PATH = BASE_DIR / "runs" / "yolov11l_aug" / "exp" / "weights" / "best.pt"
MAPPING_PATH = BASE_DIR / "configs" / "class_to_category.txt"
TEST_DIR = BASE_DIR / "yolo_dataset" / "images" / "test"
OUTPUT_CSV = BASE_DIR / "results" / "yolov11l_final.csv"

# ✅ 클래스 매핑 로드
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    class_to_category = [int(line.strip()) for line in f if line.strip()]

# ✅ 모델 로드
print(f"🔍 모델 로드: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# ✅ 예측 수행
results_list = []
annotation_id = 1

for filename in sorted(os.listdir(TEST_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = TEST_DIR / filename
    image_id = int(os.path.splitext(filename)[0])  # 예: 123.png → 123

    results = model(str(image_path))[0]

    for box in results.boxes:
        class_id = int(box.cls[0])
        category_id = class_to_category[class_id]
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_x = x1
        bbox_y = y1
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        results_list.append({
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "score": score
        })
        annotation_id += 1

# ✅ CSV 저장
os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
df = pd.DataFrame(results_list)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 결과 저장 완료: {OUTPUT_CSV}")