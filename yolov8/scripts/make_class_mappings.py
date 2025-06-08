import json
from config import MERGED_JSON_PATH, IDX2CAT_PATH, CAT2IDX_PATH

def make_class_mappings():
    with open(MERGED_JSON_PATH, "r") as f:
        data = json.load(f)

    category_ids = sorted({cat["id"] for cat in data["categories"]})
    cat2idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    idx2cat = {idx: cat_id for cat_id, idx in cat2idx.items()}

    with open(CAT2IDX_PATH, "w") as f:
        json.dump(cat2idx, f, indent=2)
    with open(IDX2CAT_PATH, "w") as f:
        json.dump(idx2cat, f, indent=2)

    print("클래스 매핑 저장 완료")
    print(f"- {CAT2IDX_PATH}")
    print(f"- {IDX2CAT_PATH}")

if __name__ == "__main__":
    make_class_mappings()