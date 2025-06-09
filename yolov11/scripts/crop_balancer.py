from pathlib import Path
import os
import json
import cv2
from collections import defaultdict
from tqdm import tqdm

# ✅ 빈 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent          # yolov11/scripts/
BASE_DIR = SCRIPT_DIR.parent                          # yolov11/
DATA_DIR = BASE_DIR.parent / "data"  # ← Project/data
ORIGINAL_DIR = DATA_DIR / "ORIGINAL"
ADD_DIR = DATA_DIR / "ADD"
OUTPUT_IMG_DIR = BASE_DIR / "crops_data" / "images"
OUTPUT_JSON_DIR = BASE_DIR / "crops_data" / "jsons"

# ✅ 파라미터 설정
TARGET_COUNT = 300
MAX_ITER = 50

# ✅ 경로 리스트
annotation_dirs = [
    {"ann": ORIGINAL_DIR / "annotations", "img": ORIGINAL_DIR / "images"},
    {"ann": ADD_DIR / "annotations", "img": ADD_DIR / "images"},
]

# ✅ 저장 폴더 생성
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# ✅ 1단계: category_id 등장 개수 초기화
category_id_counter = defaultdict(int)
print("🔍 [1단계] 카테고리 등장 횟수 초기화 중...")

for dataset in annotation_dirs:
    for file in os.listdir(dataset["ann"]):
        if not file.endswith(".json"):
            continue
        json_path = dataset["ann"] / file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for ann in data["annotations"]:
                cat_id = str(ann["category_id"])
                category_id_counter[cat_id] += 1
        except Exception as e:
            print(f"❌ 오류 발생: {json_path} - {e}")

# ✅ crop용 카운터 및 ID 초기화
crop_counter = defaultdict(int)
img_id = 100000
ann_id = 1

# ✅ 2단계: 부족한 category_id에 대해 crop
for iter_num in range(MAX_ITER):
    print(f"\n🔁 [반복 {iter_num + 1}] 부족한 category_id 채워기 시작...")

    still_short = sum(1 for v in category_id_counter.values() if v < TARGET_COUNT)
    print(f"⏳ 현재 {TARGET_COUNT}개 미만인 category 수: {still_short}")
    updated = False

    for dataset in annotation_dirs:
        for file in os.listdir(dataset["ann"]):
            if not file.endswith(".json"):
                continue
            json_path = dataset["ann"] / file
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_info = data["images"][0]
                image_path = dataset["img"] / image_info["file_name"]
                width, height = image_info["width"], image_info["height"]
                categories = {str(c["id"]): c for c in data["categories"]}

                for ann in data["annotations"]:
                    cat_id = str(ann["category_id"])
                    if category_id_counter[cat_id] >= TARGET_COUNT:
                        continue

                    img = cv2.imread(str(image_path))
                    if img is None:
                        continue

                    x, y, w, h = map(int, ann["bbox"])
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, width - x), min(h, height - y)
                    if w <= 0 or h <= 0:
                        continue

                    crop_img = img[y:y + h, x:x + w]
                    if crop_img.size == 0:
                        continue

                    crop_counter[cat_id] += 1
                    save_name = f"{cat_id}_{crop_counter[cat_id]}"
                    save_img_path = OUTPUT_IMG_DIR / f"{save_name}.png"
                    save_json_path = OUTPUT_JSON_DIR / f"{save_name}.json"
                    cv2.imwrite(str(save_img_path), crop_img)

                    json_data = {
                        "images": [{
                            "file_name": f"{save_name}.png",
                            "width": w,
                            "height": h,
                            "id": img_id
                        }],
                        "annotations": [{
                            "id": ann_id,
                            "image_id": img_id,
                            "bbox": [0, 0, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "ignore": 0,
                            "segmentation": [],
                            "category_id": int(cat_id)
                        }],
                        "categories": [categories[cat_id]] if cat_id in categories else []
                    }

                    with open(save_json_path, 'w', encoding='utf-8') as jf:
                        json.dump(json_data, jf, ensure_ascii=False, indent=2)

                    category_id_counter[cat_id] += 1
                    img_id += 1
                    ann_id += 1
                    updated = True

                    print(f"✅ crop 완료: category_id {cat_id} → {category_id_counter[cat_id]} / {TARGET_COUNT}")

            except Exception as e:
                print(f"❌ 오류: {json_path} - {e}")

    if not updated:
        print("✅ 더 이상 추가할 수 있는 항목이 없습니다. 반복 종료.")
        break

# ✅ 출력 요조
print("\n📊 참조 카테고리별 개수 요조:")
for cat_id, count in sorted(category_id_counter.items(), key=lambda x: int(x[0])):
    status = f"✅ 완료" if count >= TARGET_COUNT else f"⚠️ 부족({count}/{TARGET_COUNT})"
    print(f"  - category_id {cat_id}: {status}")