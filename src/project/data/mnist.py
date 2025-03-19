from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        num_workers: int = 1,
        pin_memory: bool = True,
        batch_size: int = 16,
        transforms: list | None = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = data_dir
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.transforms = transforms or []
        self._default_transforms = [
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ]

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            transform = T.Compose(self.transforms + self._default_transforms)

            mnist_full = MNIST(self.data_dir, train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            transform = T.Compose(self._default_transforms)

            self.mnist_test = MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            batch_size=self.batch_size,
        )
