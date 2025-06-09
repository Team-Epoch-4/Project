
from PIL import Image
import cv2
import torch

def classify_crop(
    crop_img,
    classifier_model,
    transform,
    device,
    apply_softmax=True,
    class_names=None
):
    """
    Crop 이미지를 분류 모델에 입력하여 예측 결과를 반환합니다.

    Args:
        crop_img (np.ndarray): OpenCV BGR 이미지 (YOLO crop)
        classifier_model (torch.nn.Module): PyTorch 분류 모델
        transform (torchvision.transforms): 이미지 전처리 transform
        device (torch.device): 연산에 사용할 디바이스 (cpu or cuda)
        apply_softmax (bool): True면 softmax 적용 (confidence 출력용)
        class_names (list or None): 클래스 이름 리스트 (선택)

    Returns:
        tuple: (predicted_class_id, confidence, class_name) 또는 (id, conf, None)
    """
    try:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"BGR → RGB 변환 실패: {e}")

    crop_pil = Image.fromarray(crop_rgb)
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)

    classifier_model.eval()
    with torch.no_grad():
        output = classifier_model(input_tensor)
        if apply_softmax:
            probs = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        else:
            conf, pred_idx = torch.max(output, dim=1)

    class_id = pred_idx.item()
    confidence = float(conf.item())
    class_name = class_names[class_id] if class_names else None

    return class_id, confidence, class_name
