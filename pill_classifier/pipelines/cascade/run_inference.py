
import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

from cascade.yolo_cascade_pipeline import load_yolo_model, get_cropped_bboxes
from cascade.classifier_utils import classify_crop
from Resnet18.model_utils import build_resnet18

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ResNet Cascade Inference")

    parser.add_argument('--image_path', type=str, required=True,
                        help='입력 이미지 경로')
    parser.add_argument('--yolo_weight', type=str, required=True,
                        help='YOLO 모델 weight (.pt)')
    parser.add_argument('--resnet_weight', type=str, required=True,
                        help='ResNet 모델 weight (.pth)')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                        help='YOLO confidence threshold')
    parser.add_argument('--no_plot', action='store_true',
                        help='시각화 출력 비활성화')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"❌ 이미지 파일 없음: {args.image_path}")

    # 🔹 모델 로딩
    yolo_model = load_yolo_model(args.yolo_weight)

    resnet_model = build_resnet18(num_classes=73)
    resnet_model.load_state_dict(torch.load(args.resnet_weight, map_location=device))
    resnet_model.to(device).eval()

    # 🔹 Transform 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 🔹 YOLO bbox → crop
    crops = get_cropped_bboxes(args.image_path, yolo_model, conf_thresh=args.conf_thresh, return_xyxy=True)

    for i, (crop, bbox) in enumerate(crops):
        pred_id, conf, _ = classify_crop(crop, resnet_model, transform, device)

        print(f"🧪 [{i+1}] 예측 category_id: {pred_id}, 확신도: {conf:.2%} (bbox: {bbox})")

        if not args.no_plot:
            plt.figure(figsize=(2.5, 2.5))
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.title(f"ID {pred_id} ({conf:.2%})")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
