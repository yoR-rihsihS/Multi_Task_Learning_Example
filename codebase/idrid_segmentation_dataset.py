import os
import cv2
import numpy as np

import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.transforms import v2

class IDRiDSegmentation(Dataset):
    def __init__(self, root, transform=None, mode="train"):
        assert mode in ["train", "eval"], f"mode {mode} should be either 'train' or 'eval'."
        self.file_names = []        
        self.root = root
        self.transform = transform
        self.mode = mode

        if self.mode == "train":
            self.folder_name = "a. Training Set"
        else:
            self.folder_name = "b. Testing Set"

        self.img_path = "1. Original Images/"
        self.gt_path = "2. All Segmentation Groundtruths/"
        self.subdirs = ["1. Microaneurysms", "2. Haemorrhages", "3. Hard Exudates", "4. Soft Exudates", "5. Optic Disc"]
        self.suffixes = ["_MA.tif", "_HE.tif", "_EX.tif", "_SE.tif", "_OD.tif"]
        self.task_names = ["segmentation_MA", "segmentation_HE", "segmentation_EX", "segmentation_SE", "segmentation_OD"]

        self.make_dataset()

    def make_dataset(self):
        p = os.path.join(self.root, self.img_path, self.folder_name)
        for file in os.listdir(p):
            self.file_names.append(file)

    def read_mask(self, path):
        if os.path.exists(path):
            mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            mask = np.zeros((2848, 4288), dtype=np.uint8) # Masks are all of this shape
        mask[mask != 0] = 1
        # "IDRiD_81_EX.tif" has shape (2848, 4288, 4) and the third channel is valid mask. Since, there is only one such mask, I have hard coded the following to deal with it.
        if mask.shape == (2848, 4288, 4):
            return mask[:, :, 3]
        return mask

    def get_mask(self, file_name):
        masks = []
        for i in range(5):
            p = os.path.join(self.root, self.gt_path, self.folder_name, self.subdirs[i], file_name[:-4] + self.suffixes[i])
            masks.append(self.read_mask(p))
        return np.stack(masks, axis=0) # shape: [5, 2848, 4288]
    
    def get_image(self, file_name):
        p = os.path.join(self.root, self.img_path, self.folder_name, file_name)
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img = self.get_image(file_name)
        mask = self.get_mask(file_name)
        mask = tv_tensors.Mask(mask, dtype=torch.long)
        img, mask = self.transform(img, mask)
        target = {self.task_names[i]: mask[i, :, :].unsqueeze(0) for i in range(5)}
        return img, target

def get_seg_transforms(size, norm_mean, norm_std):
    spatial_transforms_train = v2.Compose([
        v2.ToImage(),
        v2.Resize(size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=18, translate=(0.1, 0.1), scale=(0.8, 1.1)),
    ])

    spatial_transforms_val_test = v2.Compose([
        v2.ToImage(),
        v2.Resize(size),
    ])

    image_transforms_val_test = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])

    image_transforms_train = v2.Compose([
        v2.RandomPhotometricDistort(contrast=(0.9, 1.1), p=0.5),
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        ], p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])

    def transform_train(img, mask):
        img, mask = spatial_transforms_train((img, mask))
        img = image_transforms_train(img)
        return img, mask

    def transform_val_test(img, mask):
        img, mask = spatial_transforms_val_test((img, mask))
        img = image_transforms_val_test(img)
        return img, mask

    return transform_train, transform_val_test

# This is not really needed, PyTorch can auto collate.
def collate(batch):
    images, targets = [], {}
    for img, tgt in batch:
        images.append(img)
        for key, value in tgt.items():
            if key in targets:
                targets[key].append(value)
            else:
                targets[key] = [value]
    for key, value in targets.items():
        targets[key] = torch.stack(value, dim=0)
    return torch.stack(images, dim=0), targets