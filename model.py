import torch.nn as nn
import torchvision.models as models

def build_model(num_classes, in_channels=1):
    model = models.resnet18(weights=None)
    if in_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
