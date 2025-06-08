import json
import pandas as pd
from collections import Counter
from pathlib import Path
from config import MERGED_JSON_PATH, BASE_DIR

# 출력 경로
CSV_SAVE_PATH = BASE_DIR / "data/final/class_distribution.csv"
RARE_CLASS_JSON = BASE_DIR / "data/final/rare_classes.json"

# 디렉토리 생성
FINAL_DIR = BASE_DIR / "data/final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

def get_rare_classes(merged_json_path, threshold=50):
    with open(merged_json_path, "r") as f:
        data = json.load(f)

    counter = Counter()
    for ann in data["annotations"]:
        counter[int(ann["category_id"])] += 1

    # 클래스 분포 CSV 저장
    df = pd.DataFrame(sorted(counter.items()), columns=["category_id", "bbox_count"])
    df.to_csv(CSV_SAVE_PATH, index=False)
    print(f"클래스 분포 저장 완료 → {CSV_SAVE_PATH}")

    # 소수 클래스 추출
    rare_classes = [cls_id for cls_id, cnt in counter.items() if cnt < threshold]
    with open(RARE_CLASS_JSON, "w") as f:
        json.dump(rare_classes, f, indent=2)
    print(f"소수 클래스 저장 완료 → {RARE_CLASS_JSON}")
    return rare_classes

if __name__ == "__main__":
    rare_classes = get_rare_classes(MERGED_JSON_PATH)