import os
import cv2
import json
import csv

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(path: str, image):
    cv2.imwrite(path, image)

def save_json(path: str, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_csv(path: str, rows, header=None):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
