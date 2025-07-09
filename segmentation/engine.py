import torch

from .utils import compute_batch_metrics

def train_one_epoch(model, data_loader, loss_fn_1, loss_fn_2, optimizer, num_classes, scaler, device):
    running_loss = 0
    total_samples = 0
    total_intersection = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_pred_cardinality = torch.zeros(num_classes, device=device)
    total_target_cardinality = torch.zeros(num_classes, device=device)
    model.train()
    optimizer.zero_grad()
    for image, mask in data_loader:
        image = image.to(device)
        mask = mask.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred_logits = model(image)
            loss_1 = loss_fn_1(pred_logits, mask)
            loss_2 = loss_fn_2(pred_logits, mask)
            loss = loss_1 + loss_2
            running_loss += loss.item() * mask.shape[0]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_samples += mask.shape[0]

        pred_prob = torch.sigmoid(pred_logits)
        pred_mask = torch.zeros_like(pred_prob).long()
        pred_mask[pred_prob > 0.70] = 1
        intersection, union, pred_cardinality, target_cardinality = compute_batch_metrics(pred_mask, mask, num_classes)

        total_intersection += intersection
        total_union += union
        total_pred_cardinality += pred_cardinality
        total_target_cardinality += target_cardinality

    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = 2 * total_intersection / (total_pred_cardinality + total_target_cardinality + 1e-6)
    pixel_accuracy_per_class = total_intersection / (total_target_cardinality + 1e-6)

    metrics = {
        "loss": running_loss / total_samples,
        "iou_per_class": iou_per_class.tolist(),
        "dice_per_class": dice_per_class.tolist(),
        "pixel_accuracy_per_class": pixel_accuracy_per_class.tolist(),
    }

    return metrics

@torch.no_grad()
def evaluate(model, data_loader, loss_fn_1, loss_fn_2, num_classes, device):
    running_loss = 0
    total_samples = 0
    total_intersection = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_pred_cardinality = torch.zeros(num_classes, device=device)
    total_target_cardinality = torch.zeros(num_classes, device=device)
    model.eval()
    with torch.no_grad():
        for image, mask in data_loader:
            image = image.to(device)
            mask = mask.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_logits = model(image)
                loss_1 = loss_fn_1(pred_logits, mask)
                loss_2 = loss_fn_2(pred_logits, mask)
                loss = loss_1 + loss_2
                running_loss += loss.item() * mask.shape[0]
            
            total_samples += mask.shape[0]

            pred_prob = torch.sigmoid(pred_logits)
            pred_mask = torch.zeros_like(pred_prob).long()
            pred_mask[pred_prob > 0.70] = 1
            intersection, union, pred_cardinality, target_cardinality = compute_batch_metrics(pred_mask, mask, num_classes)

            total_intersection += intersection
            total_union += union
            total_pred_cardinality += pred_cardinality
            total_target_cardinality += target_cardinality

    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = 2 * total_intersection / (total_pred_cardinality + total_target_cardinality + 1e-6)
    pixel_accuracy_per_class = total_intersection / (total_target_cardinality + 1e-6)

    metrics = {
        "loss": running_loss / total_samples,
        "iou_per_class": iou_per_class.tolist(),
        "dice_per_class": dice_per_class.tolist(),
        "pixel_accuracy_per_class": pixel_accuracy_per_class.tolist(),
    }

    return metrics