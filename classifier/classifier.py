import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights

class Classifier(nn.Module):
    def __init__(self, backbone_name, num_classes_1, num_classes_2):
        super(Classifier, self).__init__()
        assert backbone_name in ["resnet50", "resnet101", "resnet152"], f"backbone_name {backbone_name} must be one of 'resnet50' or 'resnet101' or 'resnet152'"
        if backbone_name == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        elif backbone_name == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.backbone = nn.Sequential(
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

        self.classifier_1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_1),
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_2),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        return self.classifier_1(features), self.classifier_2(features)