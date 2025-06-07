import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(image_size=640):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),

        A.OneOf([
            A.RandomRotate90(p=1),
            A.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=0.01,
                rotate_limit=20,
                p=1,
                border_mode=0
            )
        ], p=0.6),

        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.2),
        A.GaussianBlur(p=0.1),

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
