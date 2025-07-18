import torch

def train_one_epoch(model, data_loader, loss_fn, optimizer, scaler, device):
    """
    Trains the given model for one complete epoch over the given dataset
    Args:
        - model (torch.nn.Module): The PyTorch model to be trained.
        - data_loader (torch.utils.data.DataLoader): The data loader for training data.
        - loss_fn (callable): The loss function used for calculating the error.
        - optimizer (torch.optim.Optimizer): The algorithm for updating model weights.
        - scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed-precision.
        - device (str): The device (e.g., 'cpu' or 'cuda' or 'cuda:5') to run on.
    Returns:
        - dict: A dictionary containing the average loss and accuracy metrics for each of the two model heads for the completed epoch.
    """
    running_loss_1 = 0
    running_loss_2 = 0
    total_correct_1 = 0
    total_correct_2 = 0
    total_samples = 0
    model.train()
    optimizer.zero_grad()
    for image, gt_class_1, gt_class_2 in data_loader:
        image = image.to(device)                # shape: [bs, 3, h, w]
        gt_class_1 = gt_class_1.to(device)      # shape: [bs,]
        gt_class_2 = gt_class_2.to(device)      # shape: [bs,]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits_1, logits_2 = model(image)                   # shape: [bs, num_classes_1], [bs, num_classes_2]
            loss_1 = loss_fn(logits_1, gt_class_1)
            loss_2 = loss_fn(logits_2, gt_class_2)
            running_loss_1 += loss_1.item() * image.shape[0]
            running_loss_2 += loss_2.item() * image.shape[0]
            loss = loss_1 + loss_2

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_samples += image.shape[0]
        pred_class_1 = torch.argmax(logits_1, dim=1)
        total_correct_1 += (pred_class_1 == gt_class_1).sum().item()
        pred_class_2 = torch.argmax(logits_2, dim=1)
        total_correct_2 += (pred_class_2 == gt_class_2).sum().item()

    metrics = {
        "rg_loss": running_loss_1 / total_samples,
        "rg_accuracy": 100 * total_correct_1 / total_samples,
        "mer_loss": running_loss_2 / total_samples,
        "mer_accuracy": 100 * total_correct_2 / total_samples,
    }

    return metrics

@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluates the given model over the given dataset
    Args:
        - model (torch.nn.Module): The PyTorch model to be trained.
        - data_loader (torch.utils.data.DataLoader): The data loader for training data.
        - loss_fn (callable): The loss function used for calculating the error.
        - device (str): The device (e.g., 'cpu' or 'cuda' or 'cuda:5') to run on.
    Returns:
        - dict: A dictionary containing the average loss and accuracy metrics for each of the two model heads for the completed epoch.
    """
    running_loss_1 = 0
    running_loss_2 = 0
    total_correct_1 = 0
    total_correct_2 = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for image, gt_class_1, gt_class_2 in data_loader:
            image = image.to(device)
            gt_class_1 = gt_class_1.to(device)
            gt_class_2 = gt_class_2.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits_1, logits_2 = model(image)
                loss_1 = loss_fn(logits_1, gt_class_1)
                running_loss_1 += loss_1.item() * image.shape[0]
                loss_2 = loss_fn(logits_2, gt_class_2)
                running_loss_2 += loss_2.item() * image.shape[0]

            total_samples += image.shape[0]
            pred_class_1 = torch.argmax(logits_1, dim=1)
            total_correct_1 += (pred_class_1 == gt_class_1).sum().item()
            pred_class_2 = torch.argmax(logits_2, dim=1)
            total_correct_2 += (pred_class_2 == gt_class_2).sum().item()
            
    metrics = {
        "rg_loss": running_loss_1 / total_samples,
        "rg_accuracy": 100 * total_correct_1 / total_samples,
        "mer_loss": running_loss_2 / total_samples,
        "mer_accuracy": 100 * total_correct_2 / total_samples,
    }

    return metrics