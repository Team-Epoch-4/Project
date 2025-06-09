import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from albumentations.pytorch import ToTensorV2


class FasterRCNNDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.image_names = df["image_name"].unique()
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        rows = self.df[self.df["image_name"] == image_name]
        img_path = Path(rows.iloc[0]["image_path"])

        if not img_path.exists():
            print(f"[경고] 이미지 파일 없음: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[경고] 이미지 로드 실패 (None): {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        labels = []
        for _, row in rows.iterrows():
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            bboxes.append([x, y, x + w, y + h])
            labels.append(int(row["label"]))

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            image = ToTensorV2()(image=image)["image"]
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": bboxes, "labels": labels}
        return image, target, image_name

def collate_fn(batch):
    return tuple(zip(*batch))
