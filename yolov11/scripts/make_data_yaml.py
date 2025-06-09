from pathlib import Path

# ✅ scripts/ 기준 → 상위 yolov11/로 이동
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ 설정파일 경로
PILL_LIST_PATH = BASE_DIR / "configs" / "pill_list.txt"

# ✅ 출력 YAML 경로
YAML_OUTPUT_PATH = BASE_DIR / "yolo_dataset" / "data.yaml"
YAML_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

# ✅ 클래스 이름 추출
class_names = []
with open(PILL_LIST_PATH, encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        if not name:
            continue
        if "-" in name:
            name = name.split("-")[0].strip()  # "이름 -숫자" 형태일 경우 이름만
        class_names.append(name)

# ✅ 절대 경로 작성
absolute_path = YAML_OUTPUT_PATH.parent.resolve()

# ✅ YAML 파일 생성
with open(YAML_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"path: {absolute_path}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names:\n")
    for name in class_names:
        f.write(f"  - {name}\n")

print(f"✅ data.yaml 생성 완료 → {YAML_OUTPUT_PATH}")