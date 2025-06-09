#!/bin/bash

echo "🚀 YOLOv11 전체 파이프라인 시작합니다."

# ✅ 현재 yolov11 디렉토리로 이동
cd "$(dirname "$0")"
echo "📂 현재 작업 디렉토리: $(pwd)"

# ✅ 1. requirements.txt 설치
echo "📦 requirements.txt 설치 중..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt 파일이 없습니다. 건너뜁니다."
fi

# ✅ 2. YOLOv11 모델 다운로드
echo "⬇️ YOLOv11 모델 다운로드 시작..."
python yolov11_model_download.py

# ✅ 3. data.yaml 생성
echo "📄 data.yaml 생성..."
python scripts/make_data_yaml.py

# ✅ 4. convert_to_yolo.py 실행
echo "🔁 YOLO 포맷으로 초기 데이터 변환..."
python scripts/convert_to_yolo.py

# ✅ 5. 클래스 불균형 분석
echo "📊 클래스 불균형 분석 시작..."
python scripts/analyze_class_imbalance.py

# ✅ 6. 부족 클래스 crop 및 증식
echo "✂️ crop_balancer 실행 중..."
python scripts/crop_balancer.py

# ✅ 7. 콜라주 이미지 생성
echo "🧩 콜라주 생성 중..."
python scripts/generate_collages.py

# ✅ 8. YOLO 증강 변환
echo "🎨 증강 및 YOLO 변환 시작..."
python scripts/convert_with_aug.py

# ✅ 9. Split_val
echo "🎨 Train -> split -> val ..."
python scripts/split_val.py

# ✅ 10.모델 학습
echo "🎨 모델 학습중..."
python scripts/train_model.py

# ✅ 11.모델검증
echo "🎨 모델검증중...""
python scripts/eval_model_aug.py


echo "✅ 전체 파이프라인 완료!"
