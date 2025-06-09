#!/bin/bash
set -e  # 에러 발생 시 종료

# ✅ 현재 스크립트의 절대 경로 (yolov11 내부)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ✅ yolov11 폴더 기준으로 이동
cd "$SCRIPT_DIR"

echo "📁 작업 디렉토리: $SCRIPT_DIR"

echo "📦 [1/9] 패키지 설치"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "⬇️ [2/9] YOLOv11 모델 다운로드"
python "$SCRIPT_DIR/scripts/yolov11_model_download.py"

echo "🗂️ [3/9] data.yaml 생성"
python "$SCRIPT_DIR/scripts/make_data_yaml.py"

echo "🔄 [4/9] 어노테이션 YOLO 형식으로 변환"
python "$SCRIPT_DIR/scripts/convert_to_yolo.py"

echo "🧪 [5/9] 검증 데이터 분리"
python "$SCRIPT_DIR/scripts/split_val.py"

echo "🧠 [6/9] YOLOv11 S/M/L 모델 학습"
python "$SCRIPT_DIR/scripts/train_models.py"

echo "📈 [7/9] 모델 학습 결과 분석"
python "$SCRIPT_DIR/scripts/analyze_yolov11_results.py"

echo "🔍 [8/9] 예측 오류 비교 (S/M/L)"
python "$SCRIPT_DIR/scripts/compare_wrong_predictions.py"

echo "🤝 [9/9] 앙상블 예측 오류 분석"
python "$SCRIPT_DIR/scripts/ensemble_wrong_predictions.py"

echo "✅ 전체 파이프라인 완료!"
