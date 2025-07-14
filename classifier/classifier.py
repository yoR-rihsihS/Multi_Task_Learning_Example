import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights

class Classifier(nn.Module):
    """
    A multi-head classifier model using a ResNet backbone.
    Args:
        - backbone_name (str): The backbone name to be used as feature extractor, can be one of 'resnet50' or 'resnet101' or 'resnet152'.
        - num_classes_1 (int): Number of output classes for the first classifier head.
        - num_classes_2 (nn.Sequential): Number of output classes for the second classifier head.
    Returns:
        - Tuple (tensor, tensor):
            - Output logits from the first classifier head of shape (batch_size, num_classes_1).
            - Output logits from the second classifier head of shape (batch_size, num_classes_2).
    """
    def __init__(self, backbone_name, num_classes_1, num_classes_2):
        super(Classifier, self).__init__()
        assert backbone_name in ["resnet50", "resnet101", "resnet152"], f"backbone_name {backbone_name} must be one of 'resnet50' or 'resnet101' or 'resnet152'"
        if backbone_name == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        elif backbone_name == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # input shape: [bs, 3, h, w]
        self.backbone = nn.Sequential(
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
            ),                          # shape: [bs, 64, h//4, w//4]
            model.layer1,               # shape: [bs, 256, h//4, w//4]
            model.layer2,               # shape: [bs, 512, h//8, w//8]
            model.layer3,               # shape: [bs, 1024, h//16, w//16]
            model.layer4,               # shape: [bs, 2048, h//32, w//32]
            model.avgpool,              # shape: [bs, 2048, 1, 1]
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
        # x.shape: [bs, 3, h, w]
        features = self.backbone(x) # shape: [bs, 2048, 1, 1]
        features = torch.flatten(features, start_dim=1) # shape: [bs, 2048]
        return self.classifier_1(features), self.classifier_2(features) # shapes: ([bs, num_classses_1], [bs, num_classes_2])