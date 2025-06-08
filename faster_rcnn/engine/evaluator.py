import torch
import pandas as pd
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def run_evaluation(model, data_loader, device, epoch=None, use_wandb=False, save_pred_df=False):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)
    metric.reset()

    pred_results = []

    category_df = pd.read_csv('faster_rcnn/data/category_df.csv')
    category_df['label'] += 1
    label_to_category_id = dict(zip(category_df['label'], category_df['category_id']))

    epoch_desc = f"(epoch={epoch+1})" if epoch is not None else "(no epoch)"
    for images, targets, image_names in tqdm(data_loader, desc=f"Evaluating {epoch_desc}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        metric.update(outputs, targets)

        #pred_df 저장할 시
        if save_pred_df:
            for i, output in enumerate(outputs):
                boxes = output['boxes'].detach().cpu().numpy()
                labels = output['labels'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()
                
                image_name = image_names[i]
                image_id = image_name.split('.')[0]

                for box, label, score in zip(boxes, labels, scores):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    category_id = label_to_category_id.get(label, -1)

                    pred_results.append({
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox_x': int(round(x_min)),
                        'bbox_y': int(round(y_min)),
                        'bbox_w': int(round(width)),
                        'bbox_h': int(round(height)),
                        'score': score
                    })

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

    if save_pred_df:
        pred_df = pd.DataFrame(pred_results)
        pred_df.to_csv('faster_rcnn_pred_df.csv', index=False)
        print(f"VAL_DATASET 예측 .csv저장: faster_rcnn_pred_df.csv ({len(pred_df)} predictions)")