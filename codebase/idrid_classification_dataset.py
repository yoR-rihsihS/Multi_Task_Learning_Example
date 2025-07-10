import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

class IDRiDClassification(Dataset):
    def __init__(self, path, size, normalize_mean, normalize_std, mode="train"):
        assert mode in ["train", "eval"], f"mode {mode} should be either 'train' or 'eval'."
        self.data_frame = pd.read_csv(path).iloc[:, :3]
        self.mode = mode
        if self.mode == "train":
            self.directory = "./data/B. Disease Grading/1. Original Images/a. Training Set/"
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size, interpolation=InterpolationMode.BILINEAR),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomAffine(degrees=18, translate=(0.1, 0.1), scale=(0.8, 1.1), interpolation=InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.directory = "./data/B. Disease Grading/1. Original Images/b. Testing Set/"
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size, interpolation=InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
            ])
        if normalize_mean is not None and normalize_std is not None:
            self.normalize = v2.Normalize(mean=normalize_mean, std=normalize_std)
        else:
            self.normalize = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_name = self.data_frame.iloc[idx, 0]
        gt_class_1 = self.data_frame.iloc[idx, 1]
        gt_class_2 = self.data_frame.iloc[idx, 2]

        img = cv2.imdecode(np.fromfile(self.directory + image_name + ".jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        if self.normalize is not None:
            img = self.normalize(img)

        target = {
            "classification_1": torch.tensor(gt_class_1, dtype=torch.long),
            "classification_2": torch.tensor(gt_class_2, dtype=torch.long),
        }

        return img, target
