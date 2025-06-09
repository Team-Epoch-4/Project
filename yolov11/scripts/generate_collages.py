import os
import cv2
import json
import numpy as np
import random
from collections import defaultdict
from pathlib import Path

# ✅ 경로 설정 (수정 버전)
SCRIPT_DIR = Path(__file__).resolve().parent        # yolov11/scripts
BASE_DIR = SCRIPT_DIR.parent                        # yolov11/
CROP_IMG_DIR = BASE_DIR / "crops_data" / "images"
CROP_JSON_DIR = BASE_DIR / "crops_data" / "jsons"
OUTPUT_IMG_DIR = BASE_DIR / "collage_images"
OUTPUT_JSON_DIR = BASE_DIR / "collage_json"

# ✅ 콜라주 설정
CANVAS_SIZE = 1280
MIN_DISTANCE = 50

# ✅ 폴더 생성
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# ✅ 클래스별 이미지 정리
class_to_images = defaultdict(list)
for fname in os.listdir(CROP_IMG_DIR):
    if fname.endswith('.png'):
        class_id = fname.split('_')[0]
        class_to_images[class_id].append(fname)

# ✅ 전체 이미지 리스트 및 셔플
all_images = [(cls, fname) for cls, fnames in class_to_images.items() for fname in fnames]
random.shuffle(all_images)
used_images = set()
collage_index = 1

# ✅ bbox 충돌 검사 함수
def is_valid_position(new_box, existing_boxes, min_distance):
    x, y, w, h = new_box
    for ex, ey, ew, eh in existing_boxes:
        if (abs(x - ex) < min_distance and abs(y - ey) < min_distance):
            return False
        if not (x + w < ex or ex + ew < x or y + h < ey or ey + eh < y):
            return False
    return True

# ✅ 콜라주 생성 반복
while True:
    available = [(cls, fname) for cls, fname in all_images if fname not in used_images]
    if len(available) < 3:
        break

    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    annotations = []
    placed_boxes = []
    used_classes = set()
    selected = []

    class_weights = {cls: len([f for f in fnames if f not in used_images]) for cls, fnames in class_to_images.items()}
    target_count = min(len(available), random.choice([3, 4]))

    while len(selected) < target_count:
        weighted_classes = [cls for cls in class_weights if class_weights[cls] > 0]
        if not weighted_classes:
            break
        weights = [class_weights[cls] for cls in weighted_classes]
        chosen_class = random.choices(weighted_classes, weights=weights, k=1)[0]

        candidates = [f for f in class_to_images[chosen_class] if f not in used_images and (chosen_class not in used_classes or len(selected) < 3)]
        if not candidates:
            class_weights[chosen_class] = 0
            continue

        chosen_img = random.choice(candidates)
        selected.append((chosen_class, chosen_img))
        used_classes.add(chosen_class)
        class_weights[chosen_class] -= 1

    if len(selected) < 3:
        break

    success = True
    for class_id, fname in selected:
        img_path = CROP_IMG_DIR / fname
        json_path = CROP_JSON_DIR / fname.replace('.png', '.json')

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ann = data['annotations'][0]
            category_id = ann['category_id']
            w, h = int(ann['bbox'][2]), int(ann['bbox'][3])
            img = cv2.imread(str(img_path))
        except Exception as e:
            print(f"❌ 오류: {fname}, {e}")
            success = False
            break

        placed = False
        for _ in range(100):
            x = random.randint(0, CANVAS_SIZE - w)
            y = random.randint(0, CANVAS_SIZE - h)
            if is_valid_position((x, y, w, h), placed_boxes, MIN_DISTANCE):
                canvas[y:y + h, x:x + w] = img
                annotations.append({
                    "bbox": [x, y, w, h],
                    "category_id": category_id
                })
                placed_boxes.append((x, y, w, h))
                used_images.add(fname)
                placed = True
                break

        if not placed:
            success = False
            break

    if not success:
        continue

    collage_name = f"collage_{collage_index}.png"
    json_name = collage_name.replace('.png', '.json')
    cv2.imwrite(str(OUTPUT_IMG_DIR / collage_name), canvas)

    json_data = {
        "images": [{
            "file_name": collage_name,
            "width": CANVAS_SIZE,
            "height": CANVAS_SIZE
        }],
        "annotations": annotations
    }

    with open(OUTPUT_JSON_DIR / json_name, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 저장됨: {collage_name} ({len(annotations)} objects)")
    collage_index += 1

print("🎉 모든 이미지 사용 완료 또는 남은 수 < 3 → 종료합니다.")