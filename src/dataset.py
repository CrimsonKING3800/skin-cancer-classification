"""
dataset.py — PyTorch Dataset for dual-input (image + tabular) skin cancer classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_transforms(image_size: int = 224, augment: bool = False):
    """
    Get image transforms for training or inference.

    Args:
        image_size: Target image size.
        augment: Whether to apply data augmentation.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


class SkinLesionDataset(Dataset):
    """
    Custom PyTorch Dataset that returns (image, tabular_features, label).

    Args:
        pil_images: List of PIL Image objects.
        targets: Tensor or array of binary labels.
        tabular_df: DataFrame of tabular features (without target/index columns).
        transform: torchvision transforms to apply to images.
    """

    def __init__(self, pil_images, targets, tabular_df, transform=None):
        self.images = pil_images
        self.targets = targets
        self.tabular = tabular_df
        self.transform = transform

        assert len(self.images) == len(self.tabular) == len(self.targets), \
            f"Length mismatch: images={len(self.images)}, tabular={len(self.tabular)}, targets={len(self.targets)}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        tabular_features = self.tabular.iloc[idx].values.astype(np.float32)
        label = self.targets[idx]

        return image, torch.tensor(tabular_features), label


def dataloader_to_numpy(dataloader):
    """
    Convert a DataLoader into numpy arrays for TensorFlow/Keras compatibility.

    Returns:
        images (np.ndarray): Shape (N, H, W, C)
        labels (np.ndarray): Shape (N,)
        tabular (np.ndarray): Shape (N, num_features)
    """
    images, tabular, labels = [], [], []
    for imgs, tab, y in dataloader:
        imgs = imgs.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        images.append(imgs.numpy())
        tabular.append(tab.numpy())
        labels.append(y.numpy())

    return np.concatenate(images), np.concatenate(labels), np.concatenate(tabular)
