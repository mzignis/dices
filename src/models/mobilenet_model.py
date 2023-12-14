import torch
import torch.nn as nn

import torchvision.models as models

n_classes = 6

# Load the pre-trained MobileNet
model = models.mobilenet_v2(pretrained=True)

# Modify the last layer for binary classification
num_ftrs = model.classifier[-1].in_features

head = nn.Sequential(
    nn.Dropout(p=0.8, inplace=False),

    nn.Linear(in_features=1280, out_features=64),
    nn.ReLU(inplace=True),
    nn.Dropout(),

    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(inplace=True),
    nn.Dropout(),

    nn.Linear(in_features=32, out_features=6, bias=True),
    nn.LogSoftmax(1)
)

model.classifier = head

print(model)

for param in model.features.parameters():
    param.requires_grad = False


