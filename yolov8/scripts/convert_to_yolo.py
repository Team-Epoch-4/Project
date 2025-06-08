import json
from pathlib import Path
import shutil
from tqdm import tqdm
from config import (
    ORIGINAL_IMG_DIR, ADD_IMG_DIR, VAL_IMG_DIR,
    VAL_ANN_DIR, MERGED_JSON_PATH,
    FINAL_IMG_TRAIN_DIR, FINAL_LBL_TRAIN_DIR,
    FINAL_IMG_VAL_DIR, FINAL_LBL_VAL_DIR,
    IDX2CAT_PATH, CAT2IDX_PATH
)

# 디렉토리 생성
FINAL_IMG_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
FINAL_LBL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
FINAL_IMG_VAL_DIR.mkdir(parents=True, exist_ok=True)
FINAL_LBL_VAL_DIR.mkdir(parents=True, exist_ok=True)

def convert_ann_to_yolo(annotations, image_info, cat2idx):
    yolo_lines = []
    w, h = image_info["width"], image_info["height"]
    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id not in cat2idx:
            continue
        cls = cat2idx[cat_id]
        x, y, bw, bh = ann["bbox"]
        cx = x + bw / 2
        cy = y + bh / 2
        yolo_lines.append(f"{cls} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
    return yolo_lines

def save_image_and_label(img_src_dir, img_name, dst_img_dir, label_path, yolo_lines):
    src_path = img_src_dir / img_name
    dst_path = dst_img_dir / img_name
    if not src_path.exists():
        print(f"이미지 없음: {src_path}")
        return
    shutil.copy(src_path, dst_path)
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

def process_train():
    with open(MERGED_JSON_PATH, "r") as f:
        merged = json.load(f)
    idx2cat = {int(k): int(v) for k, v in json.load(open(IDX2CAT_PATH)).items()}
    cat2idx = {v: k for k, v in idx2cat.items()}

    ann_map = {}
    for ann in merged["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    for img in tqdm(merged["images"], desc="🚀 Train 변환 중"):
        anns = ann_map.get(img["id"], [])
        yolo_lines = convert_ann_to_yolo(anns, img, cat2idx)
        if not yolo_lines:
            continue

        label_path = FINAL_LBL_TRAIN_DIR / img["file_name"].replace(".png", ".txt")

        found = False
        for img_dir in [ORIGINAL_IMG_DIR, ADD_IMG_DIR]:
            src_path = img_dir / img["file_name"]
            if src_path.exists():
                save_image_and_label(img_dir, img["file_name"], FINAL_IMG_TRAIN_DIR, label_path, yolo_lines)
                found = True
                break
        if not found:
            print(f"원본 이미지 없음: {img['file_name']}")

def process_val():
    cat2idx = {}
    for p in [CAT2IDX_PATH, IDX2CAT_PATH]:
        if p.exists():
            cat2idx = {int(k): int(v) for k, v in json.load(open(p)).items()}
            break

    json_files = list(Path(VAL_ANN_DIR).rglob("*.json"))
    for js in tqdm(json_files, desc="📦 Val 변환 중"):
        with open(js, "r") as f:
            data = json.load(f)
        image = data["images"][0]
        anns = data["annotations"]
        yolo_lines = convert_ann_to_yolo(anns, image, cat2idx)
        if not yolo_lines:
            continue
        save_image_and_label(VAL_IMG_DIR, image["file_name"], FINAL_IMG_VAL_DIR,
                             FINAL_LBL_VAL_DIR / image["file_name"].replace(".png", ".txt"),
                             yolo_lines)

if __name__ == "__main__":
    process_train()
    process_val()
    print("COCO → YOLO 변환 완료")