# Pill Detection & OCR Pipeline

YOLOv8과 EasyOCR를 활용한 알약 탐지 및 문자 인식 파이프라인 프로젝트입니다.

## 폴더 구조

```
pill-detection-ocr/
├── README.md
├── requirements.txt
├── config.yaml
├── yolov8_easyocr_pipeline/
│   ├── __init__.py
│   ├── detector.py
│   ├── ocr.py
│   ├── visualizer.py
│   ├── utils.py
│   └── run_pipeline.py
├── samples/
│   ├── input_images/
│   ├── cropped/
│   └── visualized/
├── results/
│   ├── ocr_results.json
│   ├── ocr_failures.csv
│   ├── visualized_by_class/
├── models/
│   └── best.pt
├── notebooks/
│   └── demo_pipeline.ipynb
└── .gitignore
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용법

1. `models/best.pt`에 YOLOv8 모델 가중치를 준비합니다.
2. `samples/input_images/`에 테스트 이미지를 넣습니다.
3. 전체 파이프라인 실행:

```bash
python yolov8_easyocr_pipeline/run_pipeline.py
```

## 결과 예시
- `results/ocr_results.json`: OCR 결과
- `results/ocr_failures.csv`: OCR 실패 목록
- `results/visualized_by_class/`: 시각화 이미지

## 참고
- YOLOv8: https://github.com/ultralytics/ultralytics
- EasyOCR: https://github.com/JaidedAI/EasyOCR 