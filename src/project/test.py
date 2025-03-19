from logging import getLogger
from pathlib import Path

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from project.config import instantiate_objects

logger = getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"


def test(cfg: DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): A configuration composed by Hydra.
    """
    logger.info(f"Loaded configuration {OmegaConf.to_yaml(cfg)}")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    assert cfg.checkpoint

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_objects(cfg.get("callbacks"))

    logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_objects(cfg.get("loggers"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    logger.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.checkpoint)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, checkpoint=cfg.checkpoint)


@hydra.main(
    version_base="1.3",
    config_path=CONFIG_DIR.as_posix(),
    config_name="test.yaml",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""

    test(cfg)


if __name__ == "__main__":
    main()
