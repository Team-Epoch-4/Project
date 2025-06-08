import json
import shutil
from pathlib import Path
import cv2
import random
from tqdm import tqdm
from config import (
    MERGED_JSON_PATH,
    FINAL_IMG_TRAIN_DIR,
    FINAL_LBL_TRAIN_DIR,
    RARE_CLASSES_PATH
)

FINAL_LBL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

def paste_bboxes():
    with open(MERGED_JSON_PATH, "r") as f:
        data = json.load(f)
    with open(RARE_CLASSES_PATH, "r") as f:
        rare_classes = json.load(f)

    image_id_to_info = {img["id"]: img for img in data["images"]}
    image_id_to_anns = {}
    for ann in data["annotations"]:
        image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    category_to_images = {cls: [] for cls in rare_classes}
    for img in data["images"]:
        anns = image_id_to_anns.get(img["id"], [])
        categories = set(ann["category_id"] for ann in anns)
        for cls in rare_classes:
            if cls in categories:
                category_to_images[cls].append((img, anns))

    print(f"bbox 복사 증강 중: {len(rare_classes)}개 클래스 대상")
    for cls in tqdm(rare_classes):
        images_with_cls = category_to_images[cls]
        for src_img, anns in images_with_cls:
            src_path = FINAL_IMG_TRAIN_DIR / src_img["file_name"]
            if not src_path.exists():
                print(f"이미지 없음: {src_path}")
                continue

            src = cv2.imread(str(src_path))
            h, w = src.shape[:2]

            # 약한 밝기 조절 (±10%)
            alpha = random.uniform(0.9, 1.1)  # 대비
            beta = random.randint(-10, 10)    # 밝기
            aug = cv2.convertScaleAbs(src, alpha=alpha, beta=beta)

            # 약한 회전 (±10도)
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # src 대신 변형된 aug 저장
            cv2.imwrite(str(new_path), aug)

            label_lines = []
            for ann in anns:
                if ann["category_id"] != cls:
                    continue
                x, y, bw, bh = ann["bbox"]
                cx, cy = x + bw / 2, y + bh / 2
                yolo_line = f"{cls} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"
                label_lines.append(yolo_line)

            if not label_lines:
                continue

            # 복사본 저장
            new_name = f"{src_img['file_name'].replace('.png', '')}_{cls}_{random.randint(100,999)}.png"
            new_path = FINAL_IMG_TRAIN_DIR / new_name
            new_lbl = FINAL_LBL_TRAIN_DIR / new_name.replace(".png", ".txt")

            shutil.copy(src_path, new_path)
            with open(new_lbl, "w") as f:
                f.write("\n".join(label_lines))

    print("bbox 단위 복사 증강 완료")

if __name__ == "__main__":
    paste_bboxes()