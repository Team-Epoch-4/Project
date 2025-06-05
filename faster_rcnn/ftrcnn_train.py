import os
import argparse
import torch
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import yaml

from engine.trainer import train_one_epoch
from engine.evaluator import run_evaluation
from dataset import FasterRCNNDataset, get_train_transform, get_val_transform, collate_fn

# --- argparse, yaml ---
parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--ckpt_dir", type=str, default="checkpoints_3", help="Directory to save checkpoints")
args = parser.parse_args()

with open("ftrcnn_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- wandb 조건부 활성화 ---
if args.use_wandb:
    import wandb
    wandb.init(project="pill-detection", name=f"fasterrcnn-{args.ckpt_dir}")
else:
    os.environ["WANDB_MODE"] = "disabled"

# --- 기본 설정 ---
EPOCHS = config["training"]["epochs"]
start_epoch = config["training"]["start_epoch"]
NUM_CLASSES = config["model"]["num_classes"]  # (배경 포함)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
os.makedirs(args.ckpt_dir, exist_ok=True)

# --- 모델 및 옵티마이저 정의 ---
model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                            lr=config["training"]["learning_rate"],
                            weight_decay=config["training"]["weight_decay"])

# --- 데이터셋/로더 정의 ---
train_df = pd.read_csv(config["data"]["train_csv"])
val_df = pd.read_csv(config["data"]["val_csv"])
image_dir = config["data"]["image_dir"]

train_dataset = FasterRCNNDataset(train_df, image_dir=image_dir, transforms=get_train_transform())
val_dataset = FasterRCNNDataset(val_df, image_dir=image_dir, transforms=get_val_transform())

train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,num_workers=0, pin_memory=True)

# --- 학습 루프 ---
for epoch in range(start_epoch, EPOCHS):
    train_one_epoch(model, optimizer, train_loader, device, epoch, use_wandb=args.use_wandb)
    run_evaluation(model, val_loader, device, epoch, use_wandb=args.use_wandb)

    # 모델 저장
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(args.ckpt_dir, f"epoch_{epoch+1:02d}.pth"))
