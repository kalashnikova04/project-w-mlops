from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torchvision
from torchvision import transforms


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        dataloader_num_wokers: int,
        batch_size: int,
        image_mean: List,
        image_std: List,
        size_h: int = 96,
        size_w: int = 96,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root_path = root_path
        self.dataloader_num_wokers = dataloader_num_wokers
        self.batch_size = batch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_h = size_h
        self.size_w = size_w

    def setup(self, stage: Optional[str] = None):

        transformer = transforms.Compose(
            [
                transforms.Resize((self.size_h, self.size_w)),
                transforms.ToTensor(),
                transforms.Normalize(self.image_mean, self.image_std),
            ]
        )

        self.train_dataset = torchvision.datasets.ImageFolder(
            Path(self.root_path, "train_11k"), transform=transformer
        )
        self.val_dataset = torchvision.datasets.ImageFolder(
            Path(self.root_path, "val"), transform=transformer
        )
        self.predict_dataset = torchvision.datasets.ImageFolder(
            Path(self.root_path, "test_labeled"), transform=transformer
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_wokers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
        )
