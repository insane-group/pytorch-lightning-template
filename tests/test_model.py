from functools import partial

import torch
from lightning import Trainer, seed_everything

from project.data.mnist import MNISTDataModule
from project.models.mnist import MNISTLitModule


def test_model() -> None:
    seed_everything(1234)

    optimizer = torch.optim.Adam
    scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.1)
    model = MNISTLitModule(optimizer=optimizer, scheduler=scheduler)
    datamodule = MNISTDataModule("../data/MNIST")
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
