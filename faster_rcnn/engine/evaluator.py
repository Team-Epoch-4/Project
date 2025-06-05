import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def run_evaluation(model, data_loader, device, epoch=None, use_wandb=False):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)
    metric.reset()

    epoch_desc = f"(epoch={epoch+1})" if epoch is not None else "(no epoch)"
    for images, targets in tqdm(data_loader, desc=f"Evaluating {epoch_desc}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        metric.update(outputs, targets)

    results = metric.compute()

    log_data = {
        f"val/{k}": (
            v.float().mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1
            else v.item() if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in results.items()
    }

    if epoch is not None:
        log_data["epoch"] = epoch

    if use_wandb:
        import wandb
        wandb.log(log_data)

    print(f"\n[Evaluation Result{' (epoch='+str(epoch+1)+')' if epoch is not None else ''}]")
    for k, v in log_data.items():
        if k != "epoch":
            print(f"{k:20s}: {v:.4f}")
