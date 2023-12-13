from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from train import BATCH_SIZE


# Path to a directory with image dataset and subfolders for training, validation and final testing
DATA_PATH = Path(Path("."), "data")
NUM_WORKERS = 4
# Image size: even though image sizes are bigger than 64, we use this to speed up training
SIZE_H = SIZE_W = 96


# Images mean and std channelwise
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]


transformer = transforms.Compose(
    [
        transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),  # normalize image data per-channel
    ]
)


def create_dataloader(dataset, transform=transformer, shuffle=False):
    loaded_dataset = torchvision.datasets.ImageFolder(
        Path(DATA_PATH, dataset), transform=transform
    )
    if dataset == "train_11k":
        shuffle = True

    return torch.utils.data.DataLoader(
        loaded_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS
    )
