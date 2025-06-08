import os
import argparse
import cv2
import torch
import pandas as pd
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from utils.vis_utils import draw_predictions

# ----- 모델 로드 함수 (evaluate.py랑 동일) -----
def load_model(checkpoint_path, num_classes, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

# ----- 이미지 전처리 -----
def prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = T.Compose([
        T.ToTensor(),
    ])(image)
    return image, image_tensor

# ----- main -----
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = load_model(args.checkpoint, args.num_classes, device)

    # category_df 로드 (id → name 매핑)
    category_df = pd.read_csv("faster_rcnn/data/category_df.csv")
    category_df['label'] += 1
    category_id_to_name = dict(zip(category_df["label"], category_df["category_id"]))

    # 이미지 리스트 가져오기
    image_files = []
    if os.path.isdir(args.input_dir):
        image_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    else:
        print("Input directory not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 이미지별 inference
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        orig_image, image_tensor = prepare_image(image_path)

        with torch.no_grad():
            predictions = model([image_tensor.to(device)])[0]

        # 시각화
        drawn_image = draw_predictions(orig_image, predictions, category_id_to_name, score_thresh=args.score_thresh)

        # 저장
        save_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_pred.png")
        cv2.imwrite(save_path, cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))
        print(f"[{image_name}] → Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--num_classes", type=int, default=74)
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="fasterrcnn_visual_results", help="Directory to save output images")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Score threshold for displaying predictions")
    args = parser.parse_args()

    main(args)