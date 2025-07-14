import torch 
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    Args:
        - alpha (None or List[float]): If not None, acts as weighting factors for the rare class: if None, loss is not scaled.
        - gamma (float): Focusing parameter to reduce the relative loss for well-classified examples. Default: 2.0.
        - size_average (bool): If True, returns the mean loss; if False, returns the sum. Default: True.
        - ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss. Default: -1.
    Returns:
        - (tensor): A scalar loss value, averaged or summed over the batch depending on 'size_average'.
    """
    def __init__(self, alpha=None, gamma=2.0, size_average=True, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs.shape: [bs, num_classes]
        # targets.shape: [bs,]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index) # shape: [bs,]
        pt = torch.exp(-1 * ce_loss) # shape: [bs,]
        if self.alpha is not None:
            at = self.alpha[targets]  # shape: [bs,]
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss # shape: [bs,]
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss # shape: [bs,]
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()