from pathlib import Path
from ultralytics import YOLO

# ✅ BASE_DIR = yolov11/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs"

# ✅ 학습할 모델 경로들
model_paths = {
    "yolov11s": MODEL_DIR / "yolo11s.pt",
    "yolov11m": MODEL_DIR / "yolo11m.pt",
    "yolov11l": MODEL_DIR / "yolo11l.pt"
}

# ✅ 학습 설정
EPOCHS = 300
PATIENCE = 5

# ✅ 모델별 학습 실행
for model_name, model_path in model_paths.items():
    print(f"\n🚀 모델 학습 시작: {model_name}")

    if not model_path.exists():
        print(f"❌ 모델 파일 없음: {model_path}")
        continue

    model = YOLO(str(model_path))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=640,
        project=str(RUNS_DIR / model_name),  # ex: yolov11/runs/yolov11s
        name="exp",
        save=True
    )

    print(f"✅ {model_name} 학습 완료")