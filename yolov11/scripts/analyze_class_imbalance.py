import os
from pathlib import Path
from collections import defaultdict

# ✅ BASE_DIR = yolov11/scripts → parent = yolov11
BASE_DIR = Path(__file__).resolve().parent
TRAIN_LABEL_DIR = BASE_DIR.parent / "yolo_dataset" / "labels" / "train"

# 클래스별 어노테이션 수 저장용
class_counts = defaultdict(int)
total_annotations = 0

# .txt 라벨 파일 순회
for file in os.listdir(TRAIN_LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    file_path = TRAIN_LABEL_DIR / file
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            class_id = line.strip().split()[0]
            class_counts[class_id] += 1
            total_annotations += 1

# ✅ 출력
print("📊 YOLO 학습 데이터 클래스 통계 (yolo_dataset/labels/train 기준):")

# 1. 클래스별 어노테이션 수 출력
for class_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f" - 클래스 {class_id}: {count}개")

# 2. 총 클래스 수
num_classes = len(class_counts)

# 3. 클래스별 평균 어노테이션 수
avg_per_class = total_annotations / num_classes if num_classes > 0 else 0

print(f"\n🔢 총 클래스 수: {num_classes}개")
print(f"🧮 총 어노테이션 수: {total_annotations}개")
print(f"📈 클래스별 평균 어노테이션 수: {avg_per_class:.2f}개")

# 4. 클래스 간 분포 편차 확인
if class_counts:
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    print(f"⚖️ 최대/최소 클래스 차이: {max_count} / {min_count} → {max_count - min_count}개 차이")