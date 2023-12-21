from pathlib import Path
from typing import Any, List

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def create_dataloader(
    root_path: str,
    dataset: str,
    size_h: int,
    size_w: int,
    image_mean: List,
    image_std: List,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader[Any]:

    transformer = transforms.Compose(
        [
            transforms.Resize((size_h, size_w)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )

    loaded_dataset = torchvision.datasets.ImageFolder(
        Path(root_path, dataset), transform=transformer
    )

    return DataLoader(
        loaded_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
