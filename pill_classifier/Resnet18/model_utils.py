import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes, pretrained=True, mode="full"):
    """
    ResNet18을 다양한 fine-tuning 방식으로 초기화하는 함수

    Parameters:
    - num_classes: 분류 클래스 수
    - pretrained: ImageNet 사전학습 여부
    - mode: ['full', 'freeze', 'partial3', 'partial4']
        - full: 전체 레이어 학습
        - freeze: 모든 backbone freeze, fc만 학습
        - partial3: layer3부터 fine-tune
        - partial4: layer4부터 fine-tune
    """
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 전체 freeze (feature extractor 모드)
    if mode == "freeze":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # layer3부터 fine-tune
    elif mode == "partial3":
        for name, param in model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # layer4부터 fine-tune
    elif mode == "partial4":
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 전체 fine-tuning
    elif mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"❌ Invalid mode: {mode}. Choose from ['full', 'freeze', 'partial3', 'partial4']")

    return model
