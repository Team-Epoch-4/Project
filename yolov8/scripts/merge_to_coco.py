# 0_merge_to_coco.py

import sys
sys.path.append("/content") 

import json
from pathlib import Path
from tqdm import tqdm
import hashlib
from config import ORIGINAL_ANN_DIR, ADD_ANN_DIR, MERGED_JSON_PATH

def get_all_json_paths(*dirs):
    json_paths = []
    for d in dirs:
        json_paths.extend(sorted(Path(d).rglob("*.json")))
    return json_paths

def hash_id(path: Path) -> int:
    return int(hashlib.sha256(str(path).encode()).hexdigest(), 16) % (10 ** 6)

def merge_jsons_to_coco(json_paths):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    categories_set = set()
    ann_id = 1

    for json_path in tqdm(json_paths, desc="JSON 병합 중"):
        with open(json_path, "r") as f:
            data = json.load(f)

        image_info = data.get("images", [])[0]
        image_path = json_path.parent.name + "/" + image_info["file_name"]
        image_id = hash_id(json_path.parent)

        coco["images"].append({
            "id": image_id,
            "file_name": image_info["file_name"],
            "width": image_info["width"],
            "height": image_info["height"]
        })

        for ann in data.get("annotations", []):
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(ann["category_id"]),
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1
            categories_set.add(int(ann["category_id"]))

    # 카테고리 정리 (id 오름차순)
    coco["categories"] = [
        {"id": cat_id, "name": str(cat_id)} for cat_id in sorted(categories_set)
    ]
    return coco

if __name__ == "__main__":
    json_paths = get_all_json_paths(ORIGINAL_ANN_DIR, ADD_ANN_DIR)
    merged = merge_jsons_to_coco(json_paths)

    with open(MERGED_JSON_PATH, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"COCO 형식으로 병합 완료 → {MERGED_JSON_PATH}")