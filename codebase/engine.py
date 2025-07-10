import torch

from .utils import compute_batch_metrics


def train_model(model, classi_loader, seg_loader, criterion, optimizer, scaler, device, iterations=20, num_classes=5, force_all_experts=True):
    s_loader = iter(seg_loader)
    c_loader = iter(classi_loader)
    
    running_loss_segmentation = 0
    running_loss_classification = 0
    total_samples_segmentation = 0
    total_samples_classification = 0
    
    total_correct_1 = 0
    total_correct_2 = 0
    total_union = torch.zeros(num_classes, device=device)
    total_intersection = torch.zeros(num_classes, device=device)
    
    total_pred_cardinality = torch.zeros(num_classes, device=device)
    total_target_cardinality = torch.zeros(num_classes, device=device)
    model.train()
    optimizer.zero_grad()
    for i in range(2 * iterations):
        if i % 2 == 0:
            try:
                images, targets = next(c_loader)
            except StopIteration:
                c_loader = iter(classi_loader)
                images, targets = next(c_loader)
        else:
            try:
                images, targets = next(s_loader)
            except StopIteration:
                s_loader = iter(seg_loader)
                images, targets = next(s_loader)

        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if i % 2 == 0:
                outputs, gate_scores = model.forward_classification(image=images, force_all_experts=force_all_experts)
            else:
                outputs, gate_scores = model.forward_segmentation(image=images, force_all_experts=force_all_experts)
            loss_dict, score_losses = criterion(outputs, targets, gate_scores)
            loss = sum(loss_dict.values()) + sum(score_losses.values())
           # if i < 2:
                # print loss_dict keys to make sure all expected losses are actually present.
               # for k, v in loss_dict.items():
                   # print(k, v)
            if i % 2 == 0:
                running_loss_classification += sum(loss_dict.values()).item() * images.shape[0]
            else: 
                running_loss_segmentation += sum(loss_dict.values()).item() * images.shape[0]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_metrics = compute_batch_metrics(outputs, targets, num_classes, device)
        
        if i % 2 == 0:
            total_samples_classification += images.shape[0]
            total_correct_1 += batch_metrics["classification_1_correct"]
            total_correct_2 += batch_metrics["classification_2_correct"]
        else:
            total_samples_segmentation += images.shape[0]
            total_intersection += batch_metrics["intersection"]
            total_union += batch_metrics["union"]
            total_pred_cardinality += batch_metrics["pred_cardinality"]
            total_target_cardinality += batch_metrics["target_cardinality"]        

    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = 2 * total_intersection / (total_pred_cardinality + total_target_cardinality + 1e-6)
    pixel_accuracy_per_class = total_intersection / (total_target_cardinality + 1e-6)

    metrics = {
        "loss_classification": running_loss_classification / total_samples_classification,
        "accuracy_1": 100 * total_correct_1 / total_samples_classification,
        "accuracy_2": 100 * total_correct_2 / total_samples_classification,
        "loss_segmentation": running_loss_segmentation / total_samples_segmentation,
        "iou_per_class": iou_per_class.tolist(),
        "dice_per_class": dice_per_class.tolist(),
        "pixel_accuracy_per_class": pixel_accuracy_per_class.tolist(),
    }

    return metrics

@torch.no_grad()
def evaluate_model(model, classi_loader, seg_loader, criterion, device, num_classes=5, force_all_experts=True):
    running_loss_classification = 0
    running_loss_segmentation = 0
    total_samples_classification = 0
    total_samples_segmentation = 0
    total_correct_1 = 0
    total_correct_2 = 0
    total_intersection = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_pred_cardinality = torch.zeros(num_classes, device=device)
    total_target_cardinality = torch.zeros(num_classes, device=device)
    model.eval()
    with torch.no_grad():
        for images, targets in classi_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, gate_scores = model(images, force_all_experts=force_all_experts)
                loss_dict, score_losses = criterion(outputs, targets, gate_scores)
                loss = sum(loss_dict.values())
                running_loss_classification += loss.item() * images.shape[0]

            batch_metrics = compute_batch_metrics(outputs, targets, num_classes, device)

            total_samples_classification += images.shape[0]
            total_correct_1 += batch_metrics["classification_1_correct"]
            total_correct_2 += batch_metrics["classification_2_correct"]

        for images, targets in seg_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, gate_scores = model(images, force_all_experts=force_all_experts)
                loss_dict, score_losses = criterion(outputs, targets, gate_scores)
                loss = sum(loss_dict.values())
                running_loss_segmentation += loss.item() * images.shape[0]

            batch_metrics = compute_batch_metrics(outputs, targets, num_classes, device)

            total_samples_segmentation += images.shape[0]
            total_intersection += batch_metrics["intersection"]
            total_union += batch_metrics["union"]
            total_pred_cardinality += batch_metrics["pred_cardinality"]
            total_target_cardinality += batch_metrics["target_cardinality"]

    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = 2 * total_intersection / (total_pred_cardinality + total_target_cardinality + 1e-6)
    pixel_accuracy_per_class = total_intersection / (total_target_cardinality + 1e-6)

    metrics = {
        "loss_classification": running_loss_classification / total_samples_classification,
        "accuracy_1": 100 * total_correct_1 / total_samples_classification,
        "accuracy_2": 100 * total_correct_2 / total_samples_classification,
        "loss_segmentation": running_loss_segmentation / total_samples_segmentation,
        "iou_per_class": iou_per_class.tolist(),
        "dice_per_class": dice_per_class.tolist(),
        "pixel_accuracy_per_class": pixel_accuracy_per_class.tolist(),
    }

    return metrics
