import os
import argparse
import torch
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import yaml

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

    # config.yaml load
    with open("faster_rcnn/ftrcnn_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 파라미터 로드
    val_df_path = config["dataset"]["processed_val_df_path"]
    num_classes = config["model"]["num_classes"]

    # 모델 로드
    model = load_model(args.checkpoint, num_classes, device)

    # Backbone profile (optional)
    if args.profile_model:
        from thop import profile
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        macs, params = profile(model.backbone, inputs=(dummy_input,))
        print(f"FLOPs (approx): {macs * 2 / 1e9:.2f} GFLOPs")
        print(f"Params: {params / 1e6:.2f} M")

    # 데이터셋 로딩
    val_df = pd.read_csv(val_df_path)
    val_dataset = FasterRCNNDataset(val_df, transforms=get_val_transform())
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # wandb 조건부
    if args.use_wandb:
        import wandb
        wandb.init(project="pill-detection", name=f"evaluate-{os.path.basename(args.checkpoint)}")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # 평가 실행
    run_evaluation(model, val_loader, device, epoch=None, use_wandb=args.use_wandb, save_pred_df=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--num_classes", type=int, default=74)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--profile_model", action="store_true", help="Profile the model backbone FLOPs and params")
    args = parser.parse_args()

    main(args)