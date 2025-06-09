# Pill_Classifier
This repository implements the classification stage of a two-stage pill detection pipeline, using ResNet on cropped images from YOLO.


# 💊 Pill Classifier Project

본 프로젝트는 **YOLO 기반 객체 탐지 모델**과 **ResNet 기반 이미지 분류 모델**을 결합하여 **알약을 정확하게 탐지하고 분류**하는 데 목적이 있습니다.  
전체 파이프라인은 다음과 같은 단계로 구성됩니다:

---

## 📁 프로젝트 구조

```bash
Pill_Classifier/
├── Dataset/                 # 데이터 전처리 스크립트
│   ├── loader_utils.py
│   ├── crop_from_yolo.py       # YOLO bbox 기반 이미지 crop
│   └── json_to_yolo_txt.py     # JSON to YOLO 포맷 변환

├── Resnet18/                # ResNet 모델 학습 코드
│   ├── model_utils.py          # ResNet18 모델 정의 (full/fine tuning 지원)
│   ├── train_utils.py          # 에폭 단위 학습/평가 함수
│   └── classifier_train_loop.py # 전체 학습 루프 (early stopping 포함)

├── pipelines/
│   ├── cascade/             # YOLO + ResNet 단순 연결 파이프라인
│   │   ├── yolo_cascade_pipeline.py  # YOLO로 crop → ResNet 분류
│   │   ├── classifier_utils.py       # crop 이미지 분류기
│   │   └── run_inference.py          # 실제 추론 실행 스크립트

│   └── ensemble/            # (향후 확장) YOLO/ResNet confidence 기반 앙상블
│       └── ensemble_pipeline.py (작성 예정)

└── README.md                # 현재 문서
