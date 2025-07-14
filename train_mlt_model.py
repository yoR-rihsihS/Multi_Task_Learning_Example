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

from codebase import MultiTaskModel, Criterion, IDRiDSegmentation, IDRiDClassification, get_seg_transforms, evaluate_model, train_model

DEVICE = 'cuda:6'
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

def save_progress(val_1_train, val_2_train, val_1_test, val_2_test, val_1_name, val_2_name, fig_name):
    plt.plot(val_1_train, label=f"{val_1_name} Train")
    plt.plot(val_2_train, label=f"{val_2_name} Train")
    plt.plot(val_1_test, label=f"{val_1_name} Test")
    plt.plot(val_2_test, label=f"{val_2_name} Test")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./saved/{fig_name}.png")
    plt.close()

def save_progress_1(val_train, val_test, val_name, fig_name):
    plt.plot(val_train, label=f"{val_name} Train")
    plt.plot(val_test, label=f"{val_name} Test")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./saved/{fig_name}.png")
    plt.close()

def print_metrics(metrics, epoch, mode):
    tasks = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"]
    print(f"Epoch {epoch} - {mode} Metrics:")
    print(f"\tClassification Loss: {metrics['loss_classification']:.04f},\tSegmentation Loss: {metrics['loss_segmentation']:.04f}")
    print(f"\tRG Classification Accuracy: {metrics['accuracy_1']:.04f},\tMER Classification Accuracy: {metrics['accuracy_2']:.04f}")
    for i in range(5):
        print(f"\t{tasks[i]} IoU: {metrics['iou_per_class'][i]:.04f},\t{tasks[i]} Dice Coefficient: {metrics['dice_per_class'][i]:.04f},\t{tasks[i]} Pixel Accuracy: {metrics['pixel_accuracy_per_class'][i]:.04f}")

def main(cfg):
    g = torch.Generator()
    g.manual_seed(SEED_VALUE)

    transform_train, transform_val_test = get_seg_transforms(cfg["size"], cfg["norm_mean"], cfg["norm_std"])
    train_seg_set = IDRiDSegmentation(root="./data/A. Segmentation/", mode="train", transform=transform_train)
    test_seg_set = IDRiDSegmentation(root="./data/A. Segmentation/", mode="eval", transform=transform_val_test)

    train_classi_set = IDRiDClassification(path=cfg["train_path"], size=cfg["size"], normalize_mean=cfg["norm_mean"], normalize_std=cfg["norm_std"], mode="train")
    test_classi_set = IDRiDClassification(path=cfg["test_path"], size=cfg["size"], normalize_mean=cfg["norm_mean"], normalize_std=cfg["norm_std"], mode="eval")

    train_seg_loader = DataLoader(train_seg_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=3, persistent_workers=True, prefetch_factor=5, worker_init_fn=seed_worker, generator=g)
    test_seg_loader = DataLoader(test_seg_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=5, worker_init_fn=seed_worker, generator=g)

    train_classi_loader = DataLoader(train_classi_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=3, persistent_workers=True, prefetch_factor=5, worker_init_fn=seed_worker, generator=g)
    test_classi_loader = DataLoader(test_classi_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=5, worker_init_fn=seed_worker, generator=g)

    model = MultiTaskModel(
        backbone=cfg["backbone"],
        task_specs=cfg["task_specs"],
        num_experts=cfg["num_experts"]
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
    scaler = GradScaler()
    criterion = Criterion(task_specs=cfg["task_specs"], alpha=0.75, gamma=4.0)

    history = {"train": [], "test": []}

    if os.path.exists(f"./saved/mlt_checkpoint.pth"):
        history = load_file(f"./saved/mlt_history.json")
        checkpoint = torch.load(f"./saved/mlt_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    use_all_experts = True

    for epoch in range(cfg["epochs"]):
        if len(history["train"]) > epoch:
            print_metrics(history["train"][epoch], epoch+1, "Train")
            print_metrics(history["test"][epoch], epoch+1, "Test")
            print()
            continue

        if epoch > 0.5 * cfg["epochs"]:
            use_all_experts = False

        train_metrics = train_model(model, train_classi_loader, train_seg_loader, criterion, optimizer, scaler, DEVICE, cfg["iterations"], 5, use_all_experts)
        print_metrics(train_metrics, epoch+1, "Train")
        test_metrics = evaluate_model(model, test_classi_loader, test_seg_loader, criterion, DEVICE, 5, use_all_experts)
        print_metrics(test_metrics, epoch+1, "Test")

        scheduler.step()

        history["train"].append(train_metrics)
        history["test"].append(test_metrics)
        print()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, f"./saved/mlt_checkpoint.pth")
        save_file(history, f"./saved/mlt_history.json")

    torch.save(model.state_dict(), f"./saved/mlt_{epoch+1}.pth")
    classi_loss_train = [h["loss_classification"] for h in history["train"]]
    classi_loss_test = [h["loss_classification"] for h in history["test"]]
    seg_loss_train = [h["loss_segmentation"] for h in history["train"]]
    seg_loss_test = [h["loss_segmentation"] for h in history["test"]]
    save_progress(classi_loss_train, classi_loss_test, seg_loss_train, seg_loss_test, "Classification Loss", "Segmentation Loss", "mlt_loss")
    rg_acc_train = [h["accuracy_1"] for h in history["train"]]
    rg_acc_test = [h["accuracy_1"] for h in history["test"]]
    mer_acc_train = [h["accuracy_2"] for h in history["train"]]
    mer_acc_test = [h["accuracy_2"] for h in history["test"]]
    save_progress(rg_acc_train, mer_acc_train, rg_acc_test, mer_acc_test, "RG Accuracy", "MER Accuracy", "mlt_acc")
    tasks = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"]
    metrics = ["iou_per_class", "dice_per_class", "pixel_accuracy_per_class"]
    name = ["IoU", "Dice Coefficent", "Pixel Accuracy"]
    for i in range(5):
        for j, metric in enumerate(metrics):
            val_train = [h[metric][i] for h in history["train"]]
            val_test = [h[metric][i] for h in history["test"]]
            save_progress_1(val_train, val_test, tasks[i]+" "+name[j], f"mlt_"+tasks[i]+"_"+name[j])
    
if __name__=="__main__":
    set_seed(SEED_VALUE)
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--backbone", type=str, required=False, default="resnet50", help="Backbone to use")
    args = parser.parse_args()
    backbone = args.backbone
    config = {
        "batch_size": 9,
        "size": [712, 1072],
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
    
        "epochs": 100,
        "iterations": 10,
        "learning_rate": 0.0003,
        "backbone_learning_rate": 0.00003,
        "weight_decay": 0.0001,
        "step": 30,
        "gamma": 0.2,

        "backbone": backbone,
        "task_specs": {
            "classification_1": 5,
            "classification_2": 3,
            "segmentation_MA": 1,
            "segmentation_HE": 1,
            "segmentation_EX": 1,
            "segmentation_SE": 1,
            "segmentation_OD": 1,
        },
        "num_experts": 3,

        "train_path": "./data/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        "test_path": "./data/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    }
    main(config)
