import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import GradScaler

from segmentation import DeepLabV3Plus, FocalLoss, train_one_epoch, evaluate, get_transforms, IDRiDSegmentation, DiceLoss

DEVICE = "cuda"
SEED_VALUE = 42
torch.cuda.empty_cache()

def set_seed(seed=SEED_VALUE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_file(history, path):
    with open(path, "w") as file:
        json.dump(history, file, indent=4)

def load_file(path):
    with open(path, "r") as file:
        history = json.load(file)
    return history

def save_progress(val_train, val_test, val_name, fig_name):
    plt.plot(val_train, label=f"{val_name} Train")
    plt.plot(val_test, label=f"{val_name} Test")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./saved/{fig_name}.png")
    plt.close()

def print_metrics(metrics, epoch, mode):
    tasks = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"]
    print(f"Epoch {epoch} - {mode} Metrics:")
    print(f"\tLoss: {metrics['loss']:.06f}")
    for i in range(5):
        print(f"\t{tasks[i]} IoU: {metrics['iou_per_class'][i]:.04f},\t{tasks[i]} Dice Coefficient: {metrics['dice_per_class'][i]:.04f},\t{tasks[i]} Pixel Accuracy: {metrics['pixel_accuracy_per_class'][i]:.04f}")

def main(cfg):
    g = torch.Generator()
    g.manual_seed(SEED_VALUE)

    transform_train, transform_val_test = get_transforms(cfg["size"], cfg["crop_size"], cfg["norm_mean"], cfg["norm_std"])
    train_set = IDRiDSegmentation(root="./data/A. Segmentation/", mode="train", transform=transform_train)
    test_set = IDRiDSegmentation(root="./data/A. Segmentation/", mode="eval", transform=transform_val_test)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=3, persistent_workers=True, prefetch_factor=5, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"]//4, shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=5, worker_init_fn=seed_worker, generator=g)

    model = DeepLabV3Plus(
        backbone=cfg["backbone"],
        num_classes=cfg["num_classes"],
        output_stride=cfg["output_stride"],
    )
    model.to(DEVICE)

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters =", num_parameters)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters =", num_parameters)

    resnet_params = list(model.resnet.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("resnet")]

    optimizer = optim.AdamW([
        {"params": resnet_params, "lr": cfg["backbone_learning_rate"]},
        {"params": other_params, "lr": cfg["learning_rate"]}
    ], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step"], gamma=cfg["gamma"])
    loss_fn_1 = FocalLoss(alpha=0.75, gamma=4.0, size_average=True)
    loss_fn_2 = DiceLoss()
    scaler = GradScaler()

    history = {"train": [], "test": []}

    if os.path.exists(f"./saved/dlv3p_os_{cfg['output_stride']}_checkpoint.pth"):
        history = load_file(f"./saved/dlv3p_os_{cfg['output_stride']}_history.json")
        checkpoint = torch.load(f"./saved/dlv3p_os_{cfg['output_stride']}_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    for epoch in range(cfg["epochs"]):
        if len(history["train"]) > epoch:
            print_metrics(history["train"][epoch], epoch+1, "Train")
            print_metrics(history["test"][epoch], epoch+1, "Test")
            print()
            continue

        train_metrics = train_one_epoch(model, train_loader, loss_fn_1, loss_fn_2, optimizer, cfg["num_classes"], scaler, DEVICE)
        print_metrics(train_metrics, epoch+1, "Train")
        test_metrics = evaluate(model, test_loader, loss_fn_1, loss_fn_2, cfg["num_classes"], DEVICE)
        print_metrics(test_metrics, epoch+1, "Test")

        scheduler.step()
        history["train"].append(train_metrics)
        history["test"].append(test_metrics)
        print()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, f"./saved/dlv3p_os_{cfg['output_stride']}_checkpoint.pth")
        save_file(history, f"./saved/dlv3p_os_{cfg['output_stride']}_history.json")

    torch.save(model.state_dict(), f"./saved/dlv3p_os_{cfg['output_stride']}_{epoch+1}.pth")
    loss_train = [h["loss"] for h in history["train"]]
    loss_test = [h["loss"] for h in history["test"]]
    save_progress(loss_train, loss_test, "Loss", f"dlv3p_os{cfg['output_stride']}"+"_loss")
    tasks = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"]
    metrics = ["iou_per_class", "dice_per_class", "pixel_accuracy_per_class"]
    name = ["IoU", "Dice Coefficent", "Pixel Accuracy"]
    for i in range(5):
        for j, metric in enumerate(metrics):
            val_train = [h[metric][i] for h in history["train"]]
            val_test = [h[metric][i] for h in history["test"]]
            save_progress(val_train, val_test, tasks[i]+" "+name[j], f"dlv3p_os{cfg['output_stride']}"+"_"+tasks[i]+"_"+name[j])
    
if __name__=="__main__":
    set_seed(SEED_VALUE)
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--backbone", type=str, required=False, default="resnet50", help="Backbone to use")
    parser.add_argument("--output_stride", type=int, required=False, default=8, help="Set output stride of backbone")
    args = parser.parse_args()
    backbone = args.backbone
    output_stride = args.output_stride
    config = {
        "batch_size": 12,
        "size": [1424, 2144],
        "crop_size": [1024, 1024],
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "backbone": backbone,
        "num_classes": 5,
        "output_stride": output_stride,
        "epochs": 100,
        "learning_rate": 0.0003,
        "backbone_learning_rate": 0.00003,
        "weight_decay": 0.0001,
        "step": 60,
        "gamma": 0.1,
    }
    main(config)