# 📦 YOLOv11 프로젝트 개요

이 프로젝트는 **YOLOv11 기반 객체 탐지 모델 학습을 위한 전체 파이프라인**을 제공합니다.  
총 **2개의 파이프라인**이 존재하며, 각 파이프라인은 목적과 처리 흐름에 따라 다음과 같이 구분됩니다.

---

## 🚀 실행 방법 (How to Run)

### 🔑 실행 전, 실행 권한 부여 (최초 1회만 수행)

```bash
chmod +x sml_eval_pipeline.sh
chmod +x augment_pipeline.sh
```

### ▶️ 파이프라인 실행
```bash
./sml_eval_pipeline.sh     # s/m/L 모델 평가용
./augment_pipeline.sh      # 데이터 증강 기반 학습용
```
### 🧪 파이프라인 1: `sml_eval_pipeline.sh`

#### ✅ 목적

YOLOv11의 사전 학습된 **세 가지 모델(s, m, l)** 을 각각 학습 및 평가합니다.  
**원본 데이터만 사용**하여 모델 성능을 비교합니다.

#### 📌 주요 작업 단계

1. 모델 다운로드 — `yolov11_model_download.py`  
2. `data.yaml` 생성 — `make_data_yaml.py`  
3. YOLO 형식 변환 — `convert_to_yolo.py`  
4. 검증 데이터 분리 — `split_val.py`  
5. 모델 학습 — `train_models.py`  
6. 성능 정리 — `analyze_yolov11_results.py`  
7. 예측 결과 비교 — `compare_wrong_predictions.py`  
8. 3개 모델의 앙상블 오답 분석 — `ensemble_wrong_predictions.py`

#### ✨ 특징

- 동일한 조건에서 **모델 크기(s/m/l)** 별 비교 가능  
- **예측 결과 분석 및 오류 클래스 비교** 가능

---

### 🧪 파이프라인 2: `augment_pipeline.sh`

#### ✅ 목적

클래스 불균형 해소 및 증강 중심의 학습 파이프라인입니다.  
`crop`, `collage`, 색상 증강을 포함하여 **YOLO 학습 데이터를 준비**합니다.

#### 📌 주요 작업 단계

1. 모델 다운로드 — `yolov11_model_download.py`  
2. `data.yaml` 생성 — `make_data_yaml.py`  
3. YOLO 형식 변환 — `convert_to_yolo.py`  
4. 클래스 분포 분석 — `analyze_class_imbalance.py`  
5. 클래스별 crop 증식 — `crop_balancer.py`  
6. collage 이미지 생성 — `generate_collages.py`  
7. YOLO 포맷 변환 + 증강 — `convert_with_aug.py`  
8. 모델 학습 — `train_model.py`  
9. 성능 평가 — `eval_model_aug.py`

#### ✨ 특징

- 클래스 불균형 해소를 위한 **crop 중심 증식**
- **random 위치 + random 클래스** 기반 collage 생성
- **Albumentations** 라이브러리를 활용한 이미지 증강
- **YOLOv11-l 단일 모델**에 집중하여 학습 수행


---

## 📁 프로젝트 구조 요약

```bash
Project/
├── yolov11/
│   ├── scripts/              # 모든 파이프라인 스크립트
│   ├── model/                # yolov11s/m/l 모델 파일
│   ├── yolo_dataset/         # YOLO 형식의 원본 이미지셋
│   ├── crops_data/           # crop된 이미지 및 jsons
│   ├── collage_images/       # collage 이미지 결과
│   ├── collage_json/         # collage json 어노테이션
│   ├── requirements.txt      # 필수 패키지 목록
│   ├── sml_eval_pipeline.sh  # s/m/l 평가용 파이프라인
│   └── augment_pipeline.sh   # 증강 중심 학습용 파이프라인
