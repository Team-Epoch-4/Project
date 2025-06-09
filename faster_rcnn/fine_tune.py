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
parser.add_argument("--resume_ckpt", type=str, default="faster_rcnn/weights/best.pth", help="Path to checkpoint to resume from")
parser.add_argument("--ckpt_dir", type=str, default="faster_rcnn/weights/finetune", help="Directory to save fine-tune checkpoints")
parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--trainable_backbone_layers", type=int, default=5, help="Number of trainable backbone layers")  # fine-tune 시 backbone 늘릴 때 사용
args = parser.parse_args()

with open("faster_rcnn/ftrcnn_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- wandb 조건부 활성화 ---
if args.use_wandb:
    import wandb
    wandb.init(project="pill-detection", name=f"fasterrcnn-{args.ckpt_dir}")
else:
    os.environ["WANDB_MODE"] = "disabled"

# --- 기본 설정 ---
FINE_TUNE_EPOCHS = config["fine_tune"]["epochs"]
save_every = config["fine_tune"]["save_every"]

NUM_CLASSES = config["model"]["num_classes"]  # (배경 포함)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
os.makedirs(args.ckpt_dir, exist_ok=True)

# --- 모델 정의 ---
model = fasterrcnn_resnet50_fpn(
    weights=None,  # head는 새로 학습
    weights_backbone='DEFAULT',  # backbone은 pretrained
    trainable_backbone_layers=args.trainable_backbone_layers,  # fine-tune 단계에서 유연하게 조절
    num_classes=NUM_CLASSES
)
model.to(device)

# --- Optimizer 정의 ---
optimizer = torch.optim.SGD(model.parameters(),
                            lr= config["fine_tune"]['learning_rate'],
                            momentum=config["fine_tune"]['momentum'],
                            weight_decay=config["fine_tune"]["weight_decay"])

# --- Checkpoint 로드 ---
print(f"Resuming from checkpoint: {args.resume_ckpt}")
checkpoint = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)  # strict=False → backbone layer 수 달라도 안전하게 로드
start_epoch = checkpoint["epoch"] + 1  # 이어서 학습

# --- Dataset / DataLoader ---
train_df = pd.read_csv(config["data"]["train_csv"])
val_df = pd.read_csv(config["data"]["val_csv"])

train_dataset = FasterRCNNDataset(train_df, transforms=get_train_transform())
val_dataset = FasterRCNNDataset(val_df, transforms=get_val_transform())

train_loader = DataLoader(train_dataset, batch_size=config["fine_tune"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

# --- 학습 루프 ---
best_map = 0.0
no_improve_count = 0
early_stop_patience = config["fine_tune"]["early_stop_patience"]
early_stop_min_delta = config["fine_tune"]["early_stop_min_delta"]


for epoch in range(start_epoch, start_epoch + FINE_TUNE_EPOCHS):
    print(f"\n[Fine-tune Epoch {epoch}/{start_epoch + FINE_TUNE_EPOCHS - 1}]")
    train_one_epoch(model, optimizer, train_loader, device, epoch, use_wandb=args.use_wandb)
    metrics = run_evaluation(model, val_loader, device, epoch, use_wandb=args.use_wandb, save_pred_df=False)

    # best.pth 저장
    val_map = metrics["val/map"]
    if val_map > best_map + early_stop_min_delta:
        best_map = val_map
        print(f"best모델 저장: mAP={val_map:.4f}, saving best.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, os.path.join(args.ckpt_dir, "best.pth"))
        no_improve_count = 0  # 개선됨 -> 카운트 리셋
    else:
        no_improve_count += 1
        print(f"Improvement 없음 -> patience counter: {no_improve_count}/{early_stop_patience}")
    # epoch별 저장
    if (epoch + 1) % save_every == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, os.path.join(args.ckpt_dir, f"epoch_{epoch+1:02d}.pth"))
    # early stop 체크
    if no_improve_count >= early_stop_patience:
        print(f"Early stopping! No improvement for {early_stop_patience} epochs.")
        break