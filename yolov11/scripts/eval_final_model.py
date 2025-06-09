from ultralytics import YOLO
from pathlib import Path
import os
import shutil

# ✅ 디렉토리 설정
SCRIPT_DIR = Path(__file__).resolve().parent          # yolov11/scripts
BASE_DIR = SCRIPT_DIR.parent                          # yolov11/
MODEL_PATH = BASE_DIR / "runs" / "yolov11l_aug" / "exp" / "weights" / "best.pt"
DATA_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"
RESULT_DIR = BASE_DIR / "final_model_metrics"

# ✅ 출력 디렉토리 정리
if RESULT_DIR.exists():
    shutil.rmtree(RESULT_DIR)
os.makedirs(RESULT_DIR, exist_ok=True)

# ✅ 모델 로드
print(f"\n📊 YOLOv11 최종 모델 평가 시작")
model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),         # val 이미지와 라벨 경로가 포함되어 있어야 함
    split="val",
    imgsz=640,
    save_json=True,
    save_hybrid=False,
    conf=0.001,
    iou=0.5,
    max_det=300,
    device=0,
    plots=True,                  # ✅ confusion matrix, PR curve, mAP 곡선 저장됨
    save_dir=str(RESULT_DIR)     # ✅ 저장 위치: final_model_metrics/
)

print(f"\n✅ 평가 완료! 결과 저장 위치: {RESULT_DIR}")