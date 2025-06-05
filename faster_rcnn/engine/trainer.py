import torch
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, use_wandb=False, log_interval=10):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1} - Training")):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        # 선택적 wandb 로깅
        if use_wandb and step % log_interval == 0:
            import wandb
            log_data = {f"train/{k}": v.item() for k, v in loss_dict.items()}
            log_data["train/total_loss"] = losses.item()
            log_data["epoch"] = epoch
            wandb.log(log_data)

    avg_loss = running_loss / len(data_loader)

    if use_wandb:
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})

    print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")
