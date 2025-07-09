import json
import numpy as np

import torch

def compute_batch_metrics(preds, targets, num_classes):
    """
    Compute metrics for a batch of predictions and targets to later compute metrics for the whole dataset.
    Args:
        - preds (torch.Tensor): The predictions of shape (N, num_classes, H, W)
        - targets (torch.Tensor): The targets of shape (N, num_classes, H, W)
        - num_classes (int): The number of classes
    Returns:
        - intersection (torch.Tensor): The intersection of the predictions and targets of shape (num_classes,)
        - union (torch.Tensor): The union of the predictions and targets of shape (num_classes,)
        - pred_cardinality (torch.Tensor): The cardinality of the predictions of shape (num_classes,)
        - target_cardinality (torch.Tensor): The cardinality of the targets of shape (num_classes,)
    """
    intersection = torch.zeros(num_classes, device=preds.device, dtype=torch.float)
    union = torch.zeros(num_classes, device=preds.device, dtype=torch.float)
    pred_cardinality = torch.zeros(num_classes, device=preds.device, dtype=torch.float)
    target_cardinality = torch.zeros(num_classes, device=preds.device, dtype=torch.float)

    for cls in range(num_classes):
        pred_mask = preds[:, cls, :, :]
        target_mask = targets[:, cls, :, :]

        inter = (pred_mask & target_mask).sum() # the intersection is same as true positive or class correct
        uni = (pred_mask | target_mask).sum()

        intersection[cls] = inter
        union[cls] = uni
        pred_cardinality[cls] = pred_mask.sum()
        target_cardinality[cls] = target_mask.sum() # the target cardinality is same as class total

    return intersection, union, pred_cardinality, target_cardinality
