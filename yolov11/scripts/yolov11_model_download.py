from pathlib import Path
from ultralytics import YOLO
import requests

# ✅ BASE_DIR 설정 (노트북도 대응)
if '__file__' in globals():
    BASE_DIR = Path(__file__).resolve().parent.parent
else:
    BASE_DIR = Path.cwd() / "yolov11"

# ✅ 모델 저장 디렉토리
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ✅ 모델 이름 및 URL 매핑
model_infos = {
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
}

available_models = []

def download_model(name: str, url: str, save_path: Path):
    print(f"\n🔽 다운로드 시도: {name}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 다운로드 완료: {save_path}")
    else:
        raise Exception(f"❌ 다운로드 실패: HTTP {response.status_code}")

# ✅ 다운로드 및 YOLO 모델 로드
for name, url in model_infos.items():
    dst_path = MODEL_DIR / name
    try:
        # 파일이 없을 때만 다운로드
        if not dst_path.exists():
            download_model(name, url, dst_path)

        # 모델 객체 생성 (로드 확인용)
        model = YOLO(str(dst_path))
        print(f"✅ 모델 로드 완료: {name}")
        available_models.append(name)

    except Exception as e:
        print(f"⚠️ 오류 발생: {name} → {e}")

# ✅ 최종 요약
print("\n📦 최종 저장된 YOLOv11 모델:")
for m in available_models:
    print(" -", m)