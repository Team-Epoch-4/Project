from pathlib import Path
from ultralytics import YOLO

# ✅ 디렉토리 설정
SCRIPT_DIR = Path(__file__).resolve().parent          # yolov11/scripts
BASE_DIR = SCRIPT_DIR.parent                          # yolov11/
MODEL_PATH = BASE_DIR / "model" / "yolo11l.pt"        # 학습할 모델 (수정 가능)
DATA_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"   # 데이터셋 yaml
RUNS_DIR = BASE_DIR / "runs" / "yolov11l_aug"         # 결과 저장 디렉토리

# ✅ 학습 설정
EPOCHS = 20
BATCH = 16
PATIENCE = 5
IMG_SIZE = 640

# ✅ 모델 로드 및 학습
print(f"🚀 모델 학습 시작: {MODEL_PATH.name}")
model = YOLO(str(MODEL_PATH))

model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    patience=PATIENCE,
    project=str(RUNS_DIR),
    name="exp",
    save=True,
    device=0,
    workers=4,
    verbose=True
)

print(f"✅ 학습 완료: {MODEL_PATH.name}")