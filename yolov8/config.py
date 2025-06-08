from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_IMG_DIR = BASE_DIR / "data/ORIGINAL/images"
ORIGINAL_ANN_DIR = BASE_DIR / "data/ORIGINAL/annotations"
ADD_IMG_DIR = BASE_DIR / "data/ADD/images"
ADD_ANN_DIR = BASE_DIR / "data/ADD/annotations"
VAL_IMG_DIR = BASE_DIR / "data/VAL/images"
VAL_ANN_DIR = BASE_DIR / "data/VAL/annotations"
TEST_IMG_DIR = BASE_DIR / "data/TEST/images"

MERGED_JSON_PATH = BASE_DIR / "data/merged.json"
FINAL_IMG_TRAIN_DIR = BASE_DIR / "data/final/images/train"
FINAL_IMG_VAL_DIR = BASE_DIR / "data/final/images/val"
FINAL_LBL_TRAIN_DIR = BASE_DIR / "data/final/labels/train"
FINAL_LBL_VAL_DIR = BASE_DIR / "data/final/labels/val"
FINAL_YAML_PATH = BASE_DIR / "data/final/data.yaml"
RARE_CLASSES_PATH = BASE_DIR / "data/final/rare_classes.json"

IDX2CAT_PATH = BASE_DIR / "data/final/idx2cat.json"
CAT2IDX_PATH = BASE_DIR / "data/final/cat2idx.json"

PRED_SAVE_PATH = BASE_DIR / "outputs/predictions"
CSV_PATH = BASE_DIR / "outputs/submission.csv"
