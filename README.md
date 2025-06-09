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

### Faster R-CNN vs YOLOv8

- **Faster R-CNN**
  - 두 단계(Two-stage) 구조, 높은 정확도
  - 느린 학습 속도, 복잡한 튜닝
      
- **YOLOv8**
  - 원스테이지(One-stage), anchor-free 구조, 빠른 속도
  - 실험 유연성, 경량화, 정확도 모두 확보

>  실험 효율성과 결과 모두에서 YOLOv8이 우수한 성능을 나타냄
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
- ResNet18 및 ResNet34 기반의 이미지 분류 실험에서도 YOLOv8과 유사한 분류 정확도를 확인
- 다만, 해당 실험은 테스트 이미지와 학습이미지 간 중복도가 높은 환경에서 진행되었기 때문에, 일반화 성능에 대한 명확한 비교에는 한계가 있음
- 따라서 이 결과는 분류기의 순수한 성능이라기보다는, **유사한 이미지 환경에서 분류 모델이 YOLO와 비슷한 성능을 낼 수 있다는 것**을 시사
  

> **결론:**  
> 실험의 효율성과 최종 성능, 실시간성까지 고려할 때, **1-stage 방식인 YOLOv8 모델을 선택하는 것이 적합**하다고 판단
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
- **YOLOv11:** 빠른 추론 속도, 간결한 구조

📌 캐글 기준 동일한 성능을 기록한 YOLOv8과 YOLOv11 모델을 **학습에 사용되지 않은 새로운 알약 이미지**로 테스트한 결과,  
**YOLOv11이 더 나은 탐지 성능과 추론 속도**를 보임 

> **결론:**
> 실험 효율성, 구조의 단순성, 일반화 성능, 그리고 추론 속도까지 종합적으로 고려했을 때,  
> **YOLOv11이 본 프로젝트의 최적 모델**로 판단되어 최종 선택되었다.
---

# 🔎 실험 결과/자료 구조

### OCR/분류기 실험
- results/ocr_results.json: OCR 결과
- results/ocr_failures.csv: OCR 실패 목록
- results/visualized_by_class/: 시각화 이미지

### YOLOv8 모델
- `final_model_metrics/` : confusion matrix, mAP 곡선
- `val_predictions/` : 예측 이미지
- `submission_yolov8n.csv` : 제출 파일

### 최종 모델
- `eval_final_model/` : confusion matrix, mAP 곡선
- `eval_model_aug/` : 예측 이미지
- `yolov11l_final.csv` : 제출 파일

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

본 프로젝트는 다양한 객체 탐지 모델(Faster R-CNN, YOLOv8, YOLOv9, YOLOv11 등)을 실험하며, 실제 알약 데이터의 특성과 성능을 종합적으로 비교하였습니다.

초기 실험에서는 YOLOv8이 빠른 추론 속도와 높은 정확도, 그리고 구조적 단순성을 바탕으로 유력한 후보 모델로 떠올랐습니다.  
특히 데이터 특성과도 잘 맞아 중간 단계에서는 YOLOv8을 기반으로 한 세부 실험(ResNet 분류기, OCR 파이프라인 비교 등)이 진행되었습니다.

그러나 실험을 확장하면서, 모델의 일반화 성능과 실전 적용 가능성을 보다 면밀히 검토하기 위해 YOLOv9, YOLOv12, YOLOv11 등 최신 모델에 대한 비교 실험을 추가로 수행하였습니다.

- YOLOv9은 구조가 복잡하고 튜닝 난이도가 높아 실험 효율성이 낮았습니다.  
- YOLOv12는 자연어 처리 기반의 실험적 모델로, 본 프로젝트 목적에 적합하지 않았습니다.  
- 반면 YOLOv11은 간결한 구조와 빠른 추론 속도를 유지하면서도, 학습에 사용되지 않은 새로운 알약 이미지에 대해 **더 높은 탐지 성능과 안정적인 일반화 성능**을 보였습니다.

> **결론적으로**, 실험 효율성, 일반화 성능, 실제 적용 가능성을 종합적으로 고려했을 때  
> 본 프로젝트의 최종 객체 탐지 모델은 **YOLOv11**로 선정되었으며, 이는 실제 의료 데이터 환경에서도 적용 가능성이 높은 실용적 모델로 판단됩니다.
