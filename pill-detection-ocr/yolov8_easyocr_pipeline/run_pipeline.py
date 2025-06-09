import os
import cv2
import yaml
from yolov8_easyocr_pipeline.detector import PillDetector
from yolov8_easyocr_pipeline.ocr import PillOCR
from yolov8_easyocr_pipeline.visualizer import draw_detections, draw_ocr_results
from yolov8_easyocr_pipeline.utils import ensure_dir, save_image, save_json, save_csv

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'samples', 'input_images')
CROPPED_DIR = os.path.join(os.path.dirname(__file__), '..', 'samples', 'cropped')
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'samples', 'visualized')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')

# Load config
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class_names = config.get('classes', ['pill'])
conf_thres = config.get('confidence_threshold', 0.25)
iou_thres = config.get('nms_iou_threshold', 0.45)
languages = config.get('ocr_languages', ['ko', 'en'])

def main():
    ensure_dir(CROPPED_DIR)
    ensure_dir(VIS_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(os.path.join(RESULTS_DIR, 'visualized_by_class'))

    detector = PillDetector(MODEL_PATH, conf_thres, iou_thres)
    ocr_engine = PillOCR(languages)

    ocr_results_all = []
    ocr_failures = []

    for fname in os.listdir(SAMPLES_DIR):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(SAMPLES_DIR, fname)
        image = cv2.imread(img_path)
        detections = detector.detect(image)
        crops = detector.crop_bboxes(image, detections)

        img_ocr_results = []
        for idx, (crop, det) in enumerate(crops):
            ocr_result = ocr_engine.read_text(crop)
            if ocr_result:
                img_ocr_results.append({
                    'det': det,
                    'ocr': ocr_result
                })
            else:
                ocr_failures.append([fname, idx, det['bbox']])
            # Save cropped image
            crop_fname = f"{os.path.splitext(fname)[0]}_crop{idx}.png"
            save_image(os.path.join(CROPPED_DIR, crop_fname), crop)

        # Save visualized image
        vis_img = draw_detections(image, detections, class_names)
        vis_img = draw_ocr_results(vis_img, [ocr for r in img_ocr_results for ocr in r['ocr']])
        save_image(os.path.join(VIS_DIR, fname), vis_img)

        ocr_results_all.append({
            'image': fname,
            'detections': img_ocr_results
        })

    # Save results
    save_json(os.path.join(RESULTS_DIR, 'ocr_results.json'), ocr_results_all)
    save_csv(os.path.join(RESULTS_DIR, 'ocr_failures.csv'), ocr_failures, header=['filename', 'crop_idx', 'bbox'])

if __name__ == '__main__':
    main()
