import os

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from project.data.mnist import MNISTDataModule
from project.models.mnist import MNISTLitModule


@pytest.fixture
def mnist_model():
    """Fixture that returns an instance of MNISTLitModule."""
    return MNISTLitModule(
        optimizer=Adam,
        scheduler=ReduceLROnPlateau,
        input_size=784,
        lin1_size=64,
        lin2_size=64,
        lin3_size=64,
        output_size=10,
        compile=False,
    )


@pytest.fixture
def mnist_batch():
    """Fixture that returns a fake batch of MNIST data."""
    batch_size = 8
    x = torch.rand(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    return x, y


def test_init(mnist_model):
    """Test that the model initializes correctly."""
    # Check that the model has the expected attributes
    assert isinstance(mnist_model.net, nn.Sequential)
    assert isinstance(mnist_model.criterion, nn.CrossEntropyLoss)
    assert mnist_model.hparams.input_size == 784
    assert mnist_model.hparams.lin1_size == 64
    assert mnist_model.hparams.lin2_size == 64
    assert mnist_model.hparams.lin3_size == 64
    assert mnist_model.hparams.output_size == 10


def test_forward(mnist_model, mnist_batch):
    """Test the forward method."""
    x, _ = mnist_batch
    output = mnist_model.forward(x)

    # Check that the output has the expected shape
    assert output.shape == (x.shape[0], mnist_model.hparams.output_size)


def test_model_step(mnist_model, mnist_batch):
    """Test the model_step method."""
    loss, preds, targets = mnist_model.model_step(mnist_batch)

    # Check that the loss is a scalar tensor
    assert loss.dim() == 0

    # Check that the predictions have the expected shape
    assert preds.shape == (mnist_batch[0].shape[0],)

    # Check that the targets match what was passed in
    assert torch.all(targets == mnist_batch[1])


def test_training_step(mnist_model, mnist_batch):
    """Test the training_step method."""
    # Reset metrics to ensure clean test
    mnist_model.train_loss.reset()
    mnist_model.train_acc.reset()

    loss = mnist_model.training_step(mnist_batch, 0)

    # Check that the loss is a scalar tensor
    assert loss.dim() == 0

    # Check that the metrics were updated
    assert mnist_model.train_loss.compute() == loss


def test_validation_step(mnist_model, mnist_batch):
    """Test the validation_step method."""
    # Reset metrics to ensure clean test
    mnist_model.val_loss.reset()
    mnist_model.val_acc.reset()

    mnist_model.validation_step(mnist_batch, 0)

    # Check that the metrics were updated
    assert mnist_model.val_loss.compute().dim() == 0
    assert 0 <= mnist_model.val_acc.compute() <= 1


def test_test_step(mnist_model, mnist_batch):
    """Test the test_step method."""
    # Reset metrics to ensure clean test
    mnist_model.test_loss.reset()
    mnist_model.test_acc.reset()

    mnist_model.test_step(mnist_batch, 0)

    # Check that the metrics were updated
    assert mnist_model.test_loss.compute().dim() == 0
    assert 0 <= mnist_model.test_acc.compute() <= 1


def test_on_validation_epoch_end(mnist_model, mnist_batch):
    """Test the on_validation_epoch_end method."""
    # Set up a val accuracy value
    mnist_model.val_acc.reset()
    mnist_model.val_acc_best.reset()

    # Simulate a validation step
    mnist_model.validation_step(mnist_batch, 0)

    # Initial accuracy value
    acc_before = mnist_model.val_acc.compute()

    # Call the hook
    mnist_model.on_validation_epoch_end()

    # Check that val_acc_best was updated
    assert mnist_model.val_acc_best.compute() == acc_before


def test_configure_optimizers(mnist_model):
    """Test the configure_optimizers method."""
    # Mock the trainer and model parameters
    mnist_model.trainer = type(
        "obj",
        (object,),
        {
            "model": type(
                "obj", (object,), {"parameters": lambda: mnist_model.parameters()}
            )
        },
    )

    optim_config = mnist_model.configure_optimizers()

    # Check that we got a dict with the right keys
    assert isinstance(optim_config, dict)
    assert "optimizer" in optim_config
    assert "lr_scheduler" in optim_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_model_cuda_compatibility(mnist_model, mnist_batch):
    """Test that the model can be moved to CUDA."""
    if torch.cuda.is_available():
        mnist_model = mnist_model.cuda()
        x, y = mnist_batch
        x, y = x.cuda(), y.cuda()

        # Test forward pass
        output = mnist_model(x)
        assert output.device.type == "cuda"

        # Test model_step
        loss, preds, targets = mnist_model.model_step((x, y))
        assert loss.device.type == "cuda"
        assert preds.device.type == "cuda"
        assert targets.device.type == "cuda"


@pytest.mark.integration
def test_integration_with_datamodule(tmp_path):
    """Integration test with actual data."""
    # Skip if running in CI without data
    if os.environ.get("CI") == "true" and not os.path.exists(tmp_path / "MNIST"):
        pytest.skip("Skipping integration test in CI without data")

    # Create data module
    dm = MNISTDataModule(
        data_dir=tmp_path,
        batch_size=4,
    )

    # Create model
    model = MNISTLitModule(
        optimizer=Adam,
        scheduler=None,
        compile=False,
    )

    # Create trainer
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        accelerator="auto",
        devices=1,
        logger=CSVLogger(save_dir=tmp_path),
        callbacks=[ModelCheckpoint(save_last=True)],
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Test
    results = trainer.test(model, datamodule=dm)

    # Check that we got results
    assert len(results) == 1
    assert "test/loss" in results[0]
    assert "test/acc" in results[0]


@pytest.mark.parametrize(
    "batch_size,lin1,lin2,lin3",
    [
        (8, 32, 32, 32),
        (16, 64, 64, 64),
        (32, 128, 128, 128),
    ],
)
def test_different_configurations(batch_size, lin1, lin2, lin3):
    """Test different model configurations."""
    model = MNISTLitModule(
        optimizer=Adam,
        scheduler=None,
        input_size=784,
        lin1_size=lin1,
        lin2_size=lin2,
        lin3_size=lin3,
        compile=False,
    )

    # Create a batch of data
    x = torch.rand(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))

    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size, 10)

    # Test model step
    loss, preds, targets = model.model_step((x, y))
    assert preds.shape == (batch_size,)
