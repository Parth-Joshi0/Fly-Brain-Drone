import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    return model