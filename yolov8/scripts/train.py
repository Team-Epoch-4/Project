from ultralytics import YOLO
from config import FINAL_YAML_PATH

def train():
    model = YOLO("yolov8n.pt")  # 다른 모델: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

    model.train(
        data=FINAL_YAML_PATH,
        epochs=50,
        imgsz=640,
        batch=16,
        name="drug_yolov8n",
        project="runs/train",
        device=0,
        workers=4
    )

if __name__ == "__main__":
    train()