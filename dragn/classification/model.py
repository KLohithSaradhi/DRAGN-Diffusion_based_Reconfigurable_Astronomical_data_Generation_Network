"""Conventional ImageNet-pretrained ResNet classifier."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .config import ResNetSection


def build_classifier(config: ResNetSection, num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
