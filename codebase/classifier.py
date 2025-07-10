import torch
import torch.nn as nn

from .additional_modules import DepthwiseSeparableConv

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()

        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 16)),
            DepthwiseSeparableConv(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_channels // 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.preprocess(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x