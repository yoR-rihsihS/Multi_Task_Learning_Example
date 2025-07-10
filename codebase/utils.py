import torch


def select_and_normalize(scores, p=0.8):
    sorted_scores, idx = torch.sort(scores, descending=True, dim=1)
    cumm_scores = torch.cumsum(sorted_scores, dim=1)

    bs, num_experts = scores.shape
    final_mask = torch.zeros_like(scores)

    for b in range(bs):
        cutoff_idx = torch.where(cumm_scores[b] >= p)[0]
        if len(cutoff_idx) == 0:
            num_selected = num_experts  # fallback to all if somehow all < p
        else:
            num_selected = cutoff_idx[0] + 1  # include this index

        selected_idx = idx[b, :num_selected]
        final_mask[b, selected_idx] = 1.0

    masked_scores = scores * final_mask
    weights = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + 1e-6)

    return final_mask, weights

def compute_batch_metrics(outputs, targets, num_classes, device):
    intersection = torch.zeros(num_classes, device=device, dtype=torch.float)
    union = torch.zeros(num_classes, device=device, dtype=torch.float)
    pred_cardinality = torch.zeros(num_classes, device=device, dtype=torch.float)
    target_cardinality = torch.zeros(num_classes, device=device, dtype=torch.float)

    task_names = ["segmentation_MA", "segmentation_HE", "segmentation_EX", "segmentation_SE", "segmentation_OD"]
    
    if "classification_1" in targets:
        pred_class_1 = torch.argmax(outputs["classification_1_logits"], dim=1)
        classification_1_correct = (pred_class_1 == targets["classification_1"]).sum().item()
    else:
        classification_1_correct = 0

    if "classification_2" in targets:
        pred_class_2 = torch.argmax(outputs["classification_2_logits"], dim=1)
        classification_2_correct = (pred_class_2 == targets["classification_2"]).sum().item()
    else:
        classification_2_correct = 0

    for i in range(num_classes):
        if task_names[i] not in targets:
            continue

        gt_mask = targets[task_names[i]]
        pred_mask_logits = outputs[task_names[i]+"_logits"]
        pred_mask = torch.zeros_like(gt_mask)
        pred_mask[pred_mask_logits > 0.5] = 1

        inter = (pred_mask & gt_mask).sum() # the intersection is same as true positive or class correct
        uni = (pred_mask | gt_mask).sum()

        intersection[i] = inter
        union[i] = uni
        pred_cardinality[i] = pred_mask.sum()
        target_cardinality[i] = gt_mask.sum() # the target cardinality is same as class total

    batch_metrics = {
        "classification_1_correct": classification_1_correct,
        "classification_2_correct": classification_2_correct,
        "intersection": intersection,
        "union": union,
        "pred_cardinality": pred_cardinality,
        "target_cardinality": target_cardinality,
    }

    return batch_metrics
