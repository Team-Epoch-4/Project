#프로젝트 구조
<!-- 
Project/
└── faster_rcnn/
    ├── train.py
    ├── evaluate.py
    ├── config.yaml
    ├── ftrcnn_requirements.txt
    │
    ├── dataset/
    │   ├── __init__.py
    │   ├── faster_rcnn_dataset.py
    │   └── transforms.py
    │
    ├── engine/
    │   ├── trainer.py
    │   └── evaluator.py
    │
    ├── checkpoints_3/
 -->

#환경 설치 gpu, ftrcnn_requirement.txt
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# pip install -r ftrcnn_requirements.txt

# faster_rcnn train_df/val_df faster_rcnn/images에서 뽑아 faster_rcnn/data/에 저장
<!-- python faster_rcnn/generate_csv.py -->

# faster_rcnn 기본 학습
<!-- python faster_rcnn/ftrcnn_train.py -->

# faster_rcnn W&B 로깅 활성화
<!-- python faster_rcnn/ftrcnn_train.py --use_wandb -->

# faster_rcnn 체크포인트 디렉토리 지정
<!-- python faster_rcnn/ftrcnn_train.py --use_wandb --ckpt_dir=checkpoints_final -->

# faster_rcnn 평가
<!-- python faster_rcnn/evaluate.py --checkpoint checkpoints_final/epoch_50.pth -->

# TEST_DATASET 시각화
<!-- python faster_rcnn/visualize_prediction.py --checkpoint checkpoints_final/epoch_50.pth --input_dir faster_rcnn/images/TEST_DATASET/images --output_dir fasterrcnn_visual_results -->