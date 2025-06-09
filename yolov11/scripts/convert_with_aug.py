import os
import json
import cv2
import albumentations as A
from pathlib import Path

def load_category_to_class_map(mapping_path):
    category_to_class = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for class_index, line in enumerate(f):
            category_id = int(line.strip())
            category_to_class[category_id] = class_index
    return category_to_class

# ✅ 증강 정의
transform = A.Compose([
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.5),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.35, hue=0.15, p=0.5),
    A.OneOf([
        A.InvertImg(p=0.15),
        A.Solarize(p=0.2),
        A.RandomToneCurve(p=0.25)
    ], p=0.25),
    A.Rotate(limit=5, border_mode=0, p=0.7),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=7),
        A.Blur(blur_limit=5)
    ], p=0.4)
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

def convert_json_folder_to_yolo_with_aug(image_dir, json_dir, output_image_dir, output_label_dir, mapping_path, target_size=640):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    category_to_class = load_category_to_class_map(mapping_path)
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_info = data['images'][0]
        img_filename = image_info['file_name']
        img_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"❌ 이미지 없음: {img_path}")
            continue

        img = cv2.imread(img_path)
        original_h, original_w = image_info['height'], image_info['width']
        scale_x = target_size / original_w
        scale_y = target_size / original_h

        bboxes, category_ids = [], []
        for ann in data['annotations']:
            x, y, w, h = ann['bbox']
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y
            bboxes.append([x, y, w, h])
            category_ids.append(int(ann['category_id']))

        if not bboxes:
            continue

        resized_img = cv2.resize(img, (target_size, target_size))

        try:
            augmented = transform(image=resized_img, bboxes=bboxes, category_ids=category_ids)
        except Exception as e:
            print(f"⚠️ 증강 오류: {img_filename} - {e}")
            continue

        image_aug = augmented['image'].astype("uint8")
        bboxes_aug = augmented['bboxes']
        category_ids_aug = augmented['category_ids']

        yolo_lines = []
        for bbox, category_id in zip(bboxes_aug, category_ids_aug):
            if category_id not in category_to_class:
                print(f"⚠️ category_id {category_id} 매핑 누락: {json_file}")
                continue

            x, y, w, h = bbox
            x_center = (x + w / 2) / target_size
            y_center = (y + h / 2) / target_size
            w_norm = w / target_size
            h_norm = h / target_size
            class_id = category_to_class[category_id]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        save_img_path = os.path.join(output_image_dir, img_filename)
        save_label_path = os.path.join(output_label_dir, img_filename.replace('.png', '.txt'))

        cv2.imwrite(save_img_path, image_aug)
        with open(save_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        print(f"✅ 변환 및 증강 완료: {img_filename}")

# ✅ 실행부
if __name__ == "__main__":
    # 현재 파일: yolov11/scripts/convert_with_aug.py
    # 기준 디렉토리: yolov11/
    BASE_DIR = Path(__file__).resolve().parent.parent

    convert_json_folder_to_yolo_with_aug(
        image_dir=BASE_DIR / "collage_images",
        json_dir=BASE_DIR / "collage_json",
        output_image_dir=BASE_DIR / "yolo_dataset" / "images" / "train",
        output_label_dir=BASE_DIR / "yolo_dataset" / "labels" / "train",
        mapping_path=BASE_DIR / "configs" / "class_to_category.txt",
        target_size=640
    )