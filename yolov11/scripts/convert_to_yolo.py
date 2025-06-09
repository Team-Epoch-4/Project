import os
import json
import cv2
from pathlib import Path

# ✅ category_id → class_id 매핑 로드
def load_category_to_class_map(mapping_path):
    category_to_class = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for class_index, line in enumerate(f):
            category_id = int(line.strip())
            category_to_class[category_id] = class_index
    return category_to_class

# ✅ YOLO 형식 변환 함수
def convert_dataset_to_yolo(image_dir, json_dir, output_image_dir, output_label_dir, category_to_class, target_size=640):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    if not image_dir.exists() or not json_dir.exists():
        print(f"❌ 경로 없음: {image_dir if not image_dir.exists() else json_dir}")
        return

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\n📂 변환 시작: {image_dir.name} ({len(image_files)}장)")

    for img_file in image_files:
        base_name = Path(img_file).stem
        json_file = base_name + ".json"

        img_path = image_dir / img_file
        json_path = json_dir / json_file

        if not json_path.exists():
            print(f"⚠️ 매칭되는 JSON 없음: {json_file}")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_info = data['images'][0]
        original_w, original_h = image_info['width'], image_info['height']

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ 이미지 로딩 실패: {img_path}")
            continue

        resized_img = cv2.resize(img, (target_size, target_size))
        scale_x = target_size / original_w
        scale_y = target_size / original_h

        yolo_lines = []
        for ann in data['annotations']:
            x, y, w, h = ann['bbox']
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y

            x_center = (x + w / 2) / target_size
            y_center = (y + h / 2) / target_size
            w_norm = w / target_size
            h_norm = h / target_size

            try:
                category_id = int(ann['category_id'])
            except Exception as e:
                print(f"❌ category_id 에러: {ann.get('category_id')} → {json_file}")
                continue

            if category_id not in category_to_class:
                print(f"⚠️ 매핑 누락 category_id {category_id} → {json_file}")
                continue

            class_id = category_to_class[category_id]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # 저장
        save_img_path = output_image_dir / img_file
        save_lbl_path = output_label_dir / (base_name + ".txt")

        cv2.imwrite(str(save_img_path), resized_img)
        with open(save_lbl_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        print(f"✅ 변환 완료: {img_file}")

# ✅ 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent          # yolov11/scripts
BASE_DIR = SCRIPT_DIR.parent                          # yolov11/
PROJECT_DIR = BASE_DIR.parent                         # Project/

MAPPING_PATH = BASE_DIR / "configs" / "class_to_category.txt"
YOLO_OUT_DIR = BASE_DIR / "yolo_dataset"
DATA_DIR = PROJECT_DIR / "data"

# ✅ 매핑 로드
category_to_class = load_category_to_class_map(MAPPING_PATH)

# ✅ 데이터셋 변환 설정 (train: ADD, ORIGINAL)
datasets = [
    {
        "name": "ADD",
        "image_dir": DATA_DIR / "ADD" / "images",
        "json_dir": DATA_DIR / "ADD" / "annotations",
        "output_type": "train"
    },
    {
        "name": "ORIGINAL",
        "image_dir": DATA_DIR / "ORIGINAL" / "images",
        "json_dir": DATA_DIR / "ORIGINAL" / "annotations",
        "output_type": "train"
    }
]

# ✅ 변환 실행
for ds in datasets:
    print(f"\n🚀 {ds['name']} 변환 중...")
    convert_dataset_to_yolo(
        image_dir=ds["image_dir"],
        json_dir=ds["json_dir"],
        output_image_dir=YOLO_OUT_DIR / "images" / ds["output_type"],
        output_label_dir=YOLO_OUT_DIR / "labels" / ds["output_type"],
        category_to_class=category_to_class
    )
