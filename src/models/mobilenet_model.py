import torch
import torch.nn as nn

import torchvision.models as models


class CustomMobileNetV2(nn.Module):

    def __init__(self, n_classes: int = None, freeze: bool = False):
        super(CustomMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.n_classes = n_classes if n_classes is not None else 6

        num_ftrs = self.model.classifier[-1].in_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.8, inplace=False),

            # num_ftrs -> 1280
            nn.Linear(in_features=num_ftrs, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(in_features=32, out_features=6, bias=True),
            nn.LogSoftmax(1)
        )
        self.model.classifier = self.head

        if freeze:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


if __name__ == '__main__':
    from src import ROOT_DIR

    is_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("mps" if is_available else "cpu")
    print(f"Using {device} device")

    # Instantiate the model
    model = CustomMobileNetV2(n_classes=6).to(device)
    model_filepath = ROOT_DIR / 'models' / 'custom-mobilenetv2.pb'
    torch.save(model, model_filepath)
    print(model)
