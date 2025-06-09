"""
📄 split_val_by_class.py

Train 디렉토리의 YOLO 라벨 데이터를 기반으로 클래스별 20%를 val 디렉토리로 분리합니다.
구조 예:
  yolov11/
  └── yolo_dataset/
      ├── images/train/
      ├── images/val/
      ├── labels/train/
      └── labels/val/
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ✅ 기준 경로: 현재 스크립트 기준으로 상위 → yolo_dataset 접근
SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"

label_dir = YOLO_DIR / "labels" / "train"
image_dir = YOLO_DIR / "images" / "train"
label_val_dir = YOLO_DIR / "labels" / "val"
image_val_dir = YOLO_DIR / "images" / "val"

os.makedirs(label_val_dir, exist_ok=True)
os.makedirs(image_val_dir, exist_ok=True)

# ✅ 이미지 경로 찾기 함수
def find_image_path(base_name, image_dir):
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = image_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None

# ✅ 클래스별 라벨 파일 수집
class_to_files = defaultdict(list)
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    path = label_dir / label_file
    with open(path, "r") as f:
        lines = f.readlines()
        if lines:
            class_id = lines[0].split()[0]
            class_to_files[class_id].append(label_file)

# ✅ 클래스별로 20%를 val로 선정
selected_for_val = set()
for class_id, file_list in class_to_files.items():
    unique_files = list(set(file_list))
    random.shuffle(unique_files)
    n_val = max(1, int(len(unique_files) * 0.2))
    selected_for_val.update(unique_files[:n_val])

# ✅ 이미지 및 라벨 이동
moved = 0
for label_file in selected_for_val:
    base_name = os.path.splitext(label_file)[0]

    label_src = label_dir / label_file
    label_dst = label_val_dir / label_file

    image_src = find_image_path(base_name, image_dir)
    image_dst = image_val_dir / image_src.name if image_src else None

    if not label_src.exists() or image_src is None:
        print(f"❌ 이동 실패: {base_name} - 파일 없음")
        continue

    shutil.move(str(label_src), str(label_dst))
    shutil.move(str(image_src), str(image_dst))
    moved += 1

print(f"✅ 이동 완료! val로 분할된 샘플 수: {moved}개")