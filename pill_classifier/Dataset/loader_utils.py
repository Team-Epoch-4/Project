import os
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

def setup_dataset_with_fixed_split(
    data_dir, transform, val_ratio=0.1, batch_size=32, seed=42,
    device='cpu', split_file=None
):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    targets = np.array([label for _, label in dataset])
    indices = np.arange(len(targets))

    # 🔹 Split 불러오기 or 생성
    if split_file and os.path.exists(split_file):
        print(f"📂 Split index 불러오는 중: {split_file}")
        saved = np.load(split_file)
        train_idx, val_idx = saved['train_idx'], saved['val_idx']
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, targets))
        if split_file:
            np.savez(split_file, train_idx=train_idx, val_idx=val_idx)
            print(f"💾 Split index 저장 완료: {split_file}")

    # 🔹 Subset 및 Dataloader 구성
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 🔹 클래스별 샘플 수 및 가중치 계산
    train_labels = [targets[i] for i in train_idx]
    class_counts = Counter(train_labels)
    num_classes = len(dataset.classes)
    class_sample_counts = [class_counts.get(i, 0) for i in range(num_classes)]

    class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    # ✅ 클래스별 통계 출력
    print("\n📊 클래스별 데이터 수 및 가중치:")
    for class_id, (cls_name, count, weight) in enumerate(zip(dataset.classes, class_sample_counts, class_weights)):
        print(f"[{class_id:02d}] {cls_name:20} | count: {count:4d} | weight: {weight:.4f}")

    return dataset.classes, train_loader, val_loader, class_weights
