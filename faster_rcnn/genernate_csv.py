from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm

def load_single_image_json(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    if not data.get("annotations") or not data.get("images"):
        return []  

    image_info = data["images"][0]
    image_name = image_info["file_name"]

    rows = []
    for ann in data["annotations"]:
        try:
            rows.append({
                "image_name": image_name,
                "image_path": str(json_path.parent.parent / "images" / image_name),
                "category_id": int(ann["category_id"]),
                "x": ann["bbox"][0],
                "y": ann["bbox"][1],
                "w": ann["bbox"][2],
                "h": ann["bbox"][3]
            })
        except Exception as e:
            print(f"{json_path.name}에서 오류 발생: {e}")
    return rows

def load_annotations_from_folder(folder_path):
    folder = Path(folder_path)
    all_jsons = list(folder.glob("*.json"))

    if not all_jsons:
        print(f"{folder_path}에서 JSON을 찾을 수 없음!")
        return pd.DataFrame()

    all_rows = []
    for json_file in tqdm(all_jsons, desc=f"Parsing {folder.name}", unit="file"):
        all_rows.extend(load_single_image_json(json_file))

    print(f"{folder.name} → 총 {len(all_rows)}개 객체 수집 완료")
    return pd.DataFrame(all_rows)

# train/val_df
train_df = load_annotations_from_folder("faster_rcnn/images/ORIGINAL_DATASET/annotations")
val_df = load_annotations_from_folder("faster_rcnn/images/TEST_DATASET/annotations")

category_df = pd.read_csv('faster_rcnn/data/category_df.csv')
category_df["label"] += 1
id2label = dict(zip(category_df["category_id"], category_df["label"]))

train_df["label"] = train_df["category_id"].map(id2label)
val_df["label"] = val_df["category_id"].map(id2label)

# float → int64로 변환
# train_df["label"] = train_df["label"].astype("int64")
# val_df["label"] = val_df["label"].astype("int64")

# 저장
Path("faster_rcnn/data").mkdir(parents=True, exist_ok=True)
train_df.to_csv("faster_rcnn/data/train_df.csv", index=False)
val_df.to_csv("faster_rcnn/data/val_df.csv", index=False)