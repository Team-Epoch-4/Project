## 사용 방법 (Usage)

이 프로젝트 폴더는 경구 약제 객체 탐지를 위한 YOLOv8 학습 파이프라인입니다 순서대로 실행해주세요

### 1️⃣ 데이터 구조 준비
`data/` 디렉토리 아래에 다음 폴더들이 있어야 합니다

```bash
data/
├── ORIGINAL/
│   ├── images/         # 원본 학습 이미지
│   └── annotations/    # COCO JSON 어노테이션 (이미지당 1개)
├── ADD/
│   ├── images/         # 추가 학습 이미지
│   └── annotations/    # COCO JSON 어노테이션
├── VAL/
│   ├── images/         # 검증 이미지
│   └── annotations/    # COCO JSON 어노테이션
├── TEST/
│   └── images/         # 테스트 이미지 (어노테이션 없음)
```

### 2️⃣ 전체 파이프라인 실행

```bash
python main.py
```

각 단계별로 수동 실행은 아래를 참고해주세요

```bash
# 0. ORIGINAL, ADD의 JSON들을 병합해 COCO 통합 어노테이션 생성
python scripts/merge_to_coco.py

# 1. 클래스별 bbox 수 분석 및 소수 클래스 자동 저장
python scripts/class_analysis.py

# 2. 통합 JSON에 등장하는 이미지들만 data/final/images/train 복사
python scripts/copy_images_from_merged_json.py

# 3. 소수 클래스 bbox만 복사 기반으로 증강 (약한 밝기/회전 포함)
python scripts/augment_rare_classes.py

# 4. idx2cat.json, cat2idx.json, data.yaml 자동 생성
python scripts/make_class_mappings.py

# 5. merged.json 및 VAL 데이터를 YOLO 포맷(.txt)으로 변환
python scripts/convert_to_yolo.py

# 6. Ultralytics YOLOv8 학습 시작 (파라미터 직접 지정 가능)
python scripts/train.py

# 7. TEST 이미지에 대해 추론 후 predictions.json 저장
python scripts/predict.py
```

## 디렉토리 구조

```bash
yolov8/
├── config.py                         # 경로 등 공용 설정
├── main.py                           # 전체 파이프라인 실행 진입점
├── README.md                         # yolov8 설명
├── scripts/                          # 각 단계별 스크립트
│   ├── merge_to_coco.py
│   ├── class_analysis.py
│   ├── copy_images_from_merged_json.py
│   ├── augment_rare_classes.py
│   ├── make_class_mappings.py
│   ├── convert_to_yolo.py
│   ├── train.py
│   └── predict.py
├── data/                             # 전체 데이터 디렉토리
│   ├── ADD/
│   │   ├── images/                   # ADD 이미지
│   │   └── annotations/             # ADD 어노테이션 (.json)
│   ├── ORIGINAL/
│   │   ├── images/                  # ORIGINAL 이미지
│   │   └── annotations/            # ORIGINAL 어노테이션 (.json)
│   ├── VAL/
│   │   ├── images/                  # 검증 이미지
│   │   └── annotations/            # 검증 어노테이션 (.json)
│   ├── TEST/
│   │   └── images/                  # 테스트 이미지 (어노테이션 없음)
│   ├── merged.json                  # 병합된 COCO 어노테이션
│   └── final/                       # 최종 YOLO 학습용 구조
│       ├── images/
│       │   ├── train/              # YOLO 학습용 이미지
│       │   └── val/                # YOLO 검증용 이미지
│       ├── labels/
│       │   ├── train/              # YOLO 학습용 라벨
│       │   └── val/                # YOLO 검증용 라벨
│       ├── data.yaml               # YOLOv8 학습용 data.yaml
│       ├── class_distribution.csv  # 클래스별 bbox 수
│       ├── rare_classes.json       # 소수 클래스 리스트
│       ├── idx2cat.json            # YOLO 인덱스 → 고유 ID
│       └── cat2idx.json            # 고유 ID → YOLO 인덱스
```

