import os
import argparse
import torch
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader

from engine.evaluator import run_evaluation
from dataset import FasterRCNNDataset, get_val_transform, collate_fn

def load_model(checkpoint_path, num_classes, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = load_model(args.checkpoint, args.num_classes, device)

    # 데이터셋 로딩
    df_val = pd.read_csv("data/val_df.csv")
    val_dataset = FasterRCNNDataset(df_val, image_dir="val_images", transforms=get_val_transform())
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # wandb 조건부
    if args.use_wandb:
        import wandb
        wandb.init(project="pill-detection", name=f"evaluate-{os.path.basename(args.checkpoint)}")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # 평가 실행
    run_evaluation(model, val_loader, device, epoch=None, use_wandb=args.use_wandb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--num_classes", type=int, default=74)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    main(args)
