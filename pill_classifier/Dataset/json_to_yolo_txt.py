

#TRAIN data>yolo

import os
import json
from shutil import copy2

# ========================================================
# 사용자 지정 경로 설정 (상황에 따라 아래 경로 수정 필요)
# ========================================================
image_dir = "/content/drive/MyDrive/초급 프로젝트/[4팀 초급 프로젝트]/data/ORIGINAL/images"  # 원본 이미지 폴더
json_dir = "/content/drive/MyDrive/초급 프로젝트/[4팀 초급 프로젝트]/data/ORIGINAL/annotations"  # 어노테이션 JSON 폴더
mapping_path = "/content/drive/MyDrive/초급 프로젝트/class_to_category.txt"  # category_id -> class_id 매핑 텍스트

output_image_dir = "/content/yolo_dataset/images/train"  # YOLO용 이미지 저장 경로
output_label_dir = "/content/yolo_dataset/labels/train"  # YOLO용 라벨(txt) 저장 경로

# ========================================================
# 경로 생성
# ========================================================
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# category_id → class_id 매핑 함수
def load_category_to_class_map(mapping_path):
    category_to_class = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for class_index, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    category_id = int(line)
                    category_to_class[category_id] = class_index
                except ValueError:
                    print(f"잘못된 category_id 무시됨: '{line}'")
    return category_to_class

category_to_class = load_category_to_class_map(mapping_path)

# JSON 파일 목록 확보 (특수문자 경로 안전)
json_paths = [
    os.path.join(json_dir, fname)
    for fname in os.listdir(json_dir)
    if fname.lower().endswith(".json")
]

print(f"변환할 JSON 수: {len(json_paths)}개")

success_count = 0

for json_path in json_paths:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_info = data.get("images", [])[0]
        file_name = image_info.get("file_name")
        width = image_info.get("width")
        height = image_info.get("height")

        if not all([file_name, width, height]):
            print(f"⚠️ 이미지 정보 누락: {json_path}")
            continue

        annotations = data.get("annotations", [])
        yolo_lines = []

        for ann in annotations:
            bbox = ann.get("bbox")
            category_id = int(ann.get("category_id", -1))

            if not bbox or category_id == -1:
                print(f"⚠️ 어노테이션 정보 누락: {json_path}")
                continue

            if category_id not in category_to_class:
                print(f"⚠️ category_id {category_id} 매핑 없음: {json_path}")
                continue

            class_id = category_to_class[category_id]
            x, y, w, h = bbox
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:
            print(f"유효한 어노테이션 없음: {json_path}")
            continue

        # 이미지 복사
        src_img_path = os.path.join(image_dir, file_name)
        dst_img_path = os.path.join(output_image_dir, file_name)

        if not os.path.exists(src_img_path):
            print(f"이미지 없음: {src_img_path}")
            continue

        copy2(src_img_path, dst_img_path)

        # 라벨 저장
        label_path = os.path.join(output_label_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_lines))

        success_count += 1
        print(f"변환 완료: {file_name}")

    except Exception as e:
        print(f"오류 발생: {json_path} → {e}")

print(f"\n변환 완료: 총 {success_count}/{len(json_paths)}개 파일")
