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

from classifier import Classifier, train_one_epoch, evaluate, FocalLoss, IDRiDClassification

DEVICE = "cuda:6"
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

def print_metrics(metrics, epoch, mode):
    print(f"Epoch {epoch} - {mode} Metrics:")
    print(f"\tRG Loss: {metrics['rg_loss']:.04f},\tRG Accuracy: {metrics['rg_accuracy']:.04f}")
    print(f"\tMER Loss: {metrics['mer_loss']:.04f},\tMER Accuracy: {metrics['mer_accuracy']:.04f}")

def main(cfg):
    model = Classifier(backbone_name=cfg["backbone"], num_classes_1=cfg["num_classes_1"], num_classes_2=cfg["num_classes_2"])
    model.to(DEVICE)

    criterion = FocalLoss(alpha=0.75, gamma=4.0, size_average=True, ignore_index=-1)

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters =", num_parameters)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters =", num_parameters)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step"], gamma=cfg["gamma"])
    scaler = GradScaler()

    g = torch.Generator()
    g.manual_seed(SEED_VALUE)

    train_set = IDRiDClassification(path=cfg["train_path"], size=cfg["size"], normalize_mean=cfg["norm_mean"], normalize_std=cfg["norm_std"], mode="train")
    test_set = IDRiDClassification(path=cfg["test_path"], size=cfg["size"], normalize_mean=cfg["norm_mean"], normalize_std=cfg["norm_std"], mode="eval")
    
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=3, prefetch_factor=10, persistent_workers=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, prefetch_factor=10, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

    history = {"train": [], "test": []}

    if os.path.exists(f"./saved/{cfg['backbone']}_checkpoint.pth"):
        history = load_file(f"./saved/{cfg['backbone']}_history.json")
        checkpoint = torch.load(f"./saved/{cfg['backbone']}_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    for epoch in range(cfg["epochs"]):
        if len(history["train"]) > epoch:
            print_metrics(history["train"][epoch], epoch+1, "Train")
            print_metrics(history["test"][epoch], epoch+1, "Test")
            print()
            continue

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        print_metrics(train_metrics, epoch+1, "Train")
        test_metrics = evaluate(model, test_loader, criterion, DEVICE)
        print_metrics(test_metrics, epoch+1, "Test")

        history["train"].append(train_metrics)
        history["test"].append(test_metrics)
        print()
        scheduler.step()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, f"./saved/{cfg['backbone']}_checkpoint.pth")
        save_file(history, f"./saved/{cfg['backbone']}_history.json")

    torch.save(model.state_dict(), f"./saved/{cfg['backbone']}_{epoch+1}.pth")
    rg_loss_train = [h["rg_loss"] for h in history["train"]]
    rg_loss_test = [h["rg_loss"] for h in history["test"]]
    mer_loss_train = [h["mer_loss"] for h in history["train"]]
    mer_loss_test = [h["mer_loss"] for h in history["test"]]
    save_progress(rg_loss_train, mer_loss_train, rg_loss_test, mer_loss_test, "RG Loss", "MER Loss", cfg["backbone"]+"_loss")
    rg_acc_train = [h["rg_accuracy"] for h in history["train"]]
    rg_acc_test = [h["rg_accuracy"] for h in history["test"]]
    mer_acc_train = [h["mer_accuracy"] for h in history["train"]]
    mer_acc_test = [h["mer_accuracy"] for h in history["test"]]
    save_progress(rg_acc_train, mer_acc_train, rg_acc_test, mer_acc_test, "RG Accuracy", "MER Accuracy", cfg["backbone"]+"_acc")

if __name__=="__main__":
    set_seed(SEED_VALUE)
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--backbone", type=str, default="resnet50", required=False, help="Model name to train")
    args = parser.parse_args()
    backbone = args.backbone
    config = {
        "backbone": backbone,
        "num_classes_1": 5,
        "num_classes_2": 3,

        "learning_rate": 0.001,
        "epochs": 30,
        "weight_decay": 0.0001,
        "gamma": 0.2,
        "step": 8,

        "batch_size": 64,
        "size": [268, 178],
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "train_path": "./data/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        "test_path": "./data/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    }
    main(config)