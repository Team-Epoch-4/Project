import os
import cv2
import argparse

def load_class_to_category_map(mapping_path):
    class_to_category = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for class_id, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    category_id = int(line)
                    class_to_category[class_id] = category_id
                except ValueError:
                    print(f"⚠️ 잘못된 category_id 무시됨: '{line}'")
    return class_to_category

def crop_yolo_bboxes_with_category(image_dir, label_dir, output_dir, mapping_path):
    os.makedirs(output_dir, exist_ok=True)
    class_to_category = load_class_to_category_map(mapping_path)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        label_file = stem + ".txt"
        label_path = os.path.join(label_dir, label_file)
        img_path = os.path.join(image_dir, img_file)

        if not os.path.exists(label_path):
            print(f"⚠️ 라벨 파일 없음: {label_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 이미지 로딩 실패: {img_path}")
            continue

        h, w, _ = image.shape
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, box_w, box_h = map(float, parts)
            class_id = int(class_id)

            # 🔁 class_id → category_id
            category_id = class_to_category.get(class_id)
            if category_id is None:
                print(f"⚠️ category_id 없음: class_id {class_id}")
                continue

            # 좌표 변환
            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cropped = image[y1:y2, x1:x2]

            # 💾 category_id 기준 저장
            category_dir = os.path.join(output_dir, f"{category_id}")
            os.makedirs(category_dir, exist_ok=True)
            save_path = os.path.join(category_dir, f"{stem}_crop_{idx}.png")
            cv2.imwrite(save_path, cropped)

        print(f"✅ 크롭 완료: {img_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 라벨 기반 크롭 (category_id 기준)")
    parser.add_argument("--image_dir", required=True, help="YOLO 이미지 폴더 경로")
    parser.add_argument("--label_dir", required=True, help="YOLO 라벨(txt) 폴더 경로")
    parser.add_argument("--output_dir", required=True, help="크롭 이미지 저장 경로")
    parser.add_argument("--mapping_path", required=True, help="class_id → category_id 매핑 파일 경로")

    args = parser.parse_args()
    crop_yolo_bboxes_with_category(args.image_dir, args.label_dir, args.output_dir, args.mapping_path)
