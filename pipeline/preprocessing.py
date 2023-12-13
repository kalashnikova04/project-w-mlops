import os

import torch
import torchvision
from torchvision import transforms
from train import BATCH_SIZE


# Path to a directory with image dataset and subfolders for training, validation and final testing
DATA_PATH = r"data"  # PATH TO THE DATASET
# Number of threads for data loader
NUM_WORKERS = 4
# Image size: even though image sizes are bigger than 64, we use this to speed up training
SIZE_H = SIZE_W = 96


# Images mean and std channelwise
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]


transformer = transforms.Compose(
    [
        transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
        transforms.ToTensor(),  # converting to tensors
        transforms.Normalize(image_mean, image_std),  # normalize image data per-channel
    ]
)


# load dataset using torchvision.datasets.ImageFolder
train_dataset = torchvision.datasets.ImageFolder(
    os.path.join(DATA_PATH, "train_11k"), transform=transformer
)
val_dataset = torchvision.datasets.ImageFolder(
    os.path.join(DATA_PATH, "val"), transform=transformer
)


# load test data also, to be used for final evaluation
test_dataset = torchvision.datasets.ImageFolder(
    os.path.join(DATA_PATH, "test_labeled"), transform=transformer
)


n_train = len(train_dataset)


train_batch_gen = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)


val_batch_gen = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)


test_batch_gen = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
