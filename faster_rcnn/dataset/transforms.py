import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(image_size=640):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),

        # 증강 요소 모두 제거 (Rotate / Shift / ColorJitter / BrightnessContrast 제거)
        # → backbone 정상 feature 추출 가능한 상태 만들기

        A.ToFloat(max_value=255.0),
        # A.Normalize(mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transform(image_size=640):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),

        A.ToFloat(max_value=255.0),
        # A.Normalize(mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))