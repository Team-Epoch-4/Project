import cv2

def draw_predictions(image, predictions, category_id_to_name, score_thresh=0.5):
    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        label_name = category_id_to_name.get(label, str(label))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label_name} {score:.2f}", (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image