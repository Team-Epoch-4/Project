프로젝트 보고서 주소 : https://smart-nightshade-63d.notion.site/20d46e2fcad88000b284c35ae13e4047?source=copy_link


# 💊 경구약제 객체 탐지 프로젝트

## 프로젝트 개요

본 프로젝트는 알약 이미지 데이터를 이용해 다양한 객체 탐지 모델을 실험하고, 실제 약제 이미지 특성에 최적화된 모델을 선정하는 데 목적이 있습니다.

**알약 객체 탐지의 주요 특성:**
- 이미지당 객체 수가 적고 겹침 없음
- 클래스 수가 많고, 시각적으로 유사한 클래스가 존재
- 각인(문자), 로고 등 다양한 분류 근거가 혼재

---

## 모델 비교 실험 및 주요 결정

### 탐지 구조 비교: Faster R-CNN vs YOLOv8

- **Faster R-CNN**
  - 두 단계(Two-stage) 구조, 높은 정확도
  - 느린 학습 속도, 복잡한 튜닝
- **YOLOv8**
  - 원스테이지(One-stage), anchor-free 구조, 빠른 속도
  - 실험 유연성, 경량화, 정확도 모두 확보

> **최종 선택:** 실험 효율성과 결과 모두에서 YOLOv8이 우수해, YOLOv8을 주력 모델로 선정.
---

### 데이터 특성과 YOLOv8 적합성

- 클래스 73개, 이미지당 객체 4개 이하
- 객체 간 거리 멀고 overlap 적음
- YOLOv8의 빠른 추론, Decoupled Head, C2f 블록 등이 특성에 적합

---

### OCR/분류기 보조 파이프라인 실험

- **OCR (Tesseract 등)**
  - 각인 미존재, 오인식, 클래스 중복 등 한계 확인
  - Dictionary 기반 후처리로도 정확도 개선 한계
- **ResNet18/34 분류기**
  - 각도, 조명, 시각적 유사성 문제로 분류 한계
  - YOLOv8 자체 분류 성능이 충분해 추가 효과 미미

> **결론:** YOLOv8의 박스/분류 결과만 사용하는 것이 가장 효율적
---

### YOLOv8n vs YOLOv8m

| Model    | 파라미터 수 | 추론 속도 | 정확도(mAP) | 학습 효율 |
|----------|-------------|-----------|-------------|-----------|
| YOLOv8n  | ~3.2M       | 매우 빠름 | 충분        | 최적      |
| YOLOv8m  | ~25M        | 중간      | 약간 ↑      | 느림      |

- 소수 객체 탐지, 반복 실험/튜닝 효율 모두에서 **YOLOv8n**이 최적

---

### YOLOv9/YOLOv12/YOLOv11 실험

- **YOLOv9:** 구조 복잡, 튜닝 난이도 ↑, 실험 효율 ↓
- **YOLOv12:** 정식 릴리즈 전, 실험적 단계
- **YOLOv11:** 빠른 추론, 유사한 정확도 → 하지만 대량 클래스에서 YOLOv8이 더 안정적

> **결론:** YOLOv8n을 최종 선택
---

# 🔎 실험 결과/자료 구조


### 모델 비교 실험
- `metrics_comparison.csv` : mAP, 파라미터 등 정량 비교
- `yolo_frcnn_visuals/` : 동일 이미지 예측 결과 비교
- `model_complexity.txt` : 파라미터/FLOPs

### YOLOv8n vs YOLOv8m
- `yolov8n_vs_m_metrics.csv` : 정량 성능 비교
- `training_curve.png` : loss/mAP 곡선
- `yolo8n_predictions/`, `yolo8m_predictions/` : 예측 결과

### OCR/분류기 실험
- `ocr_result_failures.csv` : OCR 오인식 사례
- `ocr_examples/` : 실패 이미지
- `resnet_metrics.csv`, `resnet_confusion_matrix.png` : 분류기 성능

### 최종 모델
- `final_model_metrics/` : confusion matrix, mAP 곡선
- `val_predictions/` : 예측 이미지
- `submission_yolov8n.csv` : 제출 파일

---

# 📚 참고자료

| 모델     | 링크                                         |
|----------|---------------------------------------------|
| YOLOv8   | [Ultralytics YOLOv8 공식문서](https://docs.ultralytics.com/ko/models/yolov8/)|
| Faster R-CNN | [Faster R-CNN 논문(arXiv:1504.08083)](https://arxiv.org/abs/1504.08083) |
| YOLOv9   | [Ultralytics YOLOv9 공식문서](https://docs.ultralytics.com/ko/models/yolov9/)    |
| Tesseract OCR | [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract) |
| ResNet   | [ResNet 논문(딥러닝 이미지 분류의 대표적 논문)(arXiv:1512.03385)](https://arxiv.org/abs/1506.01497)    |

---

# 💡 추가 참고

- 실험 환경 및 config: `requirements.txt`, `train_config.yaml`
- 주요 에러/이슈: `error_log.md`
- 데이터 증강 전략: Albumentations 적용

---

# 🏆 결론

- **YOLOv8n** 모델이 약제 객체 탐지에 최적화된 모델로 선정
- OCR/분류기 등 후처리 없이 YOLOv8n만으로 안정적 성능 달성
