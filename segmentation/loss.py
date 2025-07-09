import torch 
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)  # [B, 5, H, W]
        intersection = (p * targets).sum(dim=(2, 3))  # [B, 5]
        union = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # [B, 5]
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        targets = targets.float()
        loss = sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction="none")
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()