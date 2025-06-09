# Faster R-CNN Pill Detection

## 📌 프로젝트 개요
- 알약 이미지 데이터셋 기반 Faster R-CNN 객체 탐지 모델 학습/평가
- 데이터 가공 → 모델 학습 → Fine-tuning → 평가 → 예측 결과 시각화

## 프로젝트 구조
```plaintext
faster_rcnn/
├── evaluate.py                   # 모델 평가 스크립트
├── fine_tune.py                  # fine-tuning (backbone full train) 스크립트
├── ftrcnn_config.yaml            # 학습/평가 설정 파일
├── ftrcnn_train.py               # 초기 학습 스크립트 (backbone 3 layer fine-tune)
├── generate_csv.py               # 데이터셋 전처리 (train_df.csv, val_df.csv 생성)
├── README.md                     # 프로젝트 설명 파일
├── requirements.txt              # 패키지 목록
├── visualize_prediction.py       # 테스트 이미지 결과 시각화
│
├── data/
│   ├── category_df.csv
│   ├── train_df.csv
│   ├── val_df.csv
│
├── dataset/
│   ├── faster_rcnn_dataset.py
│   ├── transforms.py
│   ├── __init__.py
│
├── engine/
│   ├── evaluator.py
│   ├── trainer.py
```
---
# 환경 설치 gpu, ftrcnn_requirement.txt
## PyTorch GPU 버전 설치
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
## 프로젝트 requirements 설치
pip install -r ftrcnn_requirements.txt
---
# Workflow 예시
1. generate_csv.py → CSV 생성
2. ftrcnn_train.py → 초기 학습
3. fine_tune.py → Backbone 전체 Fine-tune
4. evaluate.py → 성능 평가
5. visualize_prediction.py → 결과 시각화

# 데이터셋 준비
- 원본 데이터를 data/ 폴더에 구성
- generate_csv.py 실행
##  생성 코드
python faster_rcnn/generate_csv.py
---
##  생성 결과:
faster_rcnn/data/train_df.csv
faster_rcnn/data/val_df.csv

## CSV 구성:
image_name, image_path, boundingbox, label
---

# faster_rcnn 기본 학습(Backbone Layer = 3)
python faster_rcnn/ftrcnn_train.py
- Pretrained COCO weight 사용
- Best model → weights/best.pth 저장
- Epoch 단위 체크포인트 → weights/epoch_XX.pth 저장
- Early-stopping 적용
## faster_rcnn W&B 로깅 활성화
python faster_rcnn/ftrcnn_train.py --use_wandb
## faster_rcnn 체크포인트 디렉토리 지정
python faster_rcnn/ftrcnn_train.py --use_wandb --ckpt_dir=faster_rcnn/weight
## Fine-tuning (Backbone Layer = 5)
python faster_rcnn/fine_tune.py --resume_ckpt faster_rcnn/weights/best.pth --use_wandb
- Backbone full-train 수행
## faster_rcnn 평가
python faster_rcnn/evaluate.py --checkpoint faster_rcnn/weights/fine_tune/best.pth
- engine/evaluator.py → run_evaluation 사용
- mAP, mAR 계산
- GFLOPs 계산 기능 포함

# TEST_DATASET 시각화
python faster_rcnn/visualize_prediction.py \
    --checkpoint faster_rcnn/weights/best.pth \
    --input_dir data/TEST/test_images \
    --output_dir fasterrcnn_visual_results

- input_dir: 테스트 이미지 폴더 경로
- output_dir: 결과 bbox 시각화 이미지 저장 경로

학습 시 다음 전략 사용:
- Early-stopping
- COCO pre-trained backbone
- layer freeze/unfreeze 전략 사용
평가 지표: 
- mAP@0.5
- mAP@0.75
- mAR@100
데이터 구성: 
- 73 클래스
- 최대 4 bbox per image