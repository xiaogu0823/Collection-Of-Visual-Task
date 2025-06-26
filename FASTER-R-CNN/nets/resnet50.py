import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights


def resnet50(pretrained=True):
    weights = None
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    features = nn.Sequential(*list(model.children())[:-3])
    classifier = nn.Sequential(*list(model.children())[-3:-1])
    return features, classifier
