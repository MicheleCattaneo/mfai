import torch
from lightning.pytorch.cli import ArgsType, LightningCLI
import pytest
import lightning.pytorch as L
import datetime

from mfai.torch.dummy_dataset import DummyDataModule
from mfai.torch.models import UNet
from mfai.torch.segmentation_module import SegmentationLightningModule
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


@pytest.mark.parametrize(
    "config",
    [
        ("binary", 1, 1),
        ("multiclass", 3, 3),
        ("multilabel", 2, 4),
        ("regression", 2, 1),
    ]
)
@pytest.mark.parametrize(
    "logger_cls",
    [
        MLFlowLogger,
        TensorBoardLogger,
    ]
)
def test_lightning_training_loop(config, logger_cls):
    """
    Checks that our lightning module is trainable in all 4 modes.
    """
    IMG_SIZE = 64
    task, in_channels, out_channels = config
    arch = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        input_shape=[IMG_SIZE, IMG_SIZE],
    )

    loss = torch.nn.CrossEntropyLoss() if task == "multiclass" else torch.nn.MSELoss()
    model = SegmentationLightningModule(arch, task, loss)

    datamodule = DummyDataModule(task, 2, IMG_SIZE, IMG_SIZE, in_channels, out_channels)
    datamodule.setup()

    # Define logger, callbacks and lightning Trainer
    logger_args = {
        TensorBoardLogger: {"save_dir": "logs/"},
        MLFlowLogger: {"experiment_name": f"test_experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    }
    logger = logger_cls(**logger_args[logger_cls])
    
    checkpointer = ModelCheckpoint(
        monitor="val_loss",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = L.Trainer(
        logger=logger,
        max_epochs=1,
        callbacks=[checkpointer],
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
    )

    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    trainer.test(model, datamodule.test_dataloader(), ckpt_path="best")


def cli_main(args: ArgsType = None):
    cli = LightningCLI(
        SegmentationLightningModule, DummyDataModule, args=args, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


def test_cli():
    cli_main(
        [
            "--model.model=Segformer",
            "--model.type_segmentation=binary",
            "--model.loss=torch.nn.BCEWithLogitsLoss",
            "--model.model.in_channels=2",
            "--model.model.out_channels=1",
            "--model.model.input_shape=[64, 64]",
            "--optimizer=AdamW",
            "--trainer.fast_dev_run=True",
        ]
    )


def test_cli_with_config_file():
    cli_main(["--config=mfai/config/cli_fit_test.yaml", "--trainer.fast_dev_run=True"])
