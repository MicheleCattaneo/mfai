from pathlib import Path
from typing import Callable, Literal, Tuple, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
import torchmetrics as tm
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers.logger import Logger
import warnings

from mfai.torch.models.base import ModelABC
from mlflow.tracking.client import MlflowClient
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from mfai.torch.padding import pad_batch, undo_padding

# define custom scalar in tensorboard, to have 2 lines on same graph
layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
    },
}

class AgnosticLogger(Logger):
    
    def __init__(self, backend_logger):
        self.backend_logger = backend_logger
        
        if type(backend_logger) not in self.supported_loggers:
            raise ValueError(f'Logger {type(backend_logger)} not yet supported.')
        
    @property
    def name(self):
        return "AgnosticLogger"
    
    @property
    def experiment(self):
        return self.backend_logger.experiment
    
    def add_custom_scalars(self, layout):
        if isinstance(self, TensorBoardLogger):
            self.backend_logger.experiment.add_custom_scalars(layout) 
    
    @property
    def run_id(self):
        if type(self.backend_logger) == MLFlowLogger:
            return self.backend_logger.run_id
        return None
    
    @property
    def supported_loggers(self):
        return MLFlowLogger, TensorBoardLogger
    
    @property
    def version(self):
        return self.backend_logger.version
    
    @property
    def log_dir(self):
        if isinstance(self.backend_logger, MLFlowLogger):
            return f"{self.backend_logger._tracking_uri.replace('file:', '')}/{self.backend_logger.experiment_id}/{self.backend_logger.run_id}"
        if isinstance(self.backend_logger, TensorBoardLogger):
            return self.backend_logger.log_dir
    
    @rank_zero_only
    def log_hyperparams(self, params, *args):
        if isinstance(self.backend_logger, MLFlowLogger):
            params_ = {**params}
            for arg in args:
                params_ = {**params_, **arg}
            self.backend_logger.log_hyperparams(params_)
        elif isinstance(self.backend_logger, TensorBoardLogger):
            self.backend_logger.log_hyperparams(params, *args)
        
    
    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.backend_logger.log_metrics(metrics, step)
        
    def log_image(self, key, image, step, **kwargs):
        if isinstance(self.backend_logger, TensorBoardLogger):
            dataformats = kwargs.get('dataformats', 'HW')
            self.backend_logger.experiment.add_image(key, image, step, dataformats=dataformats)
        elif isinstance(self.backend_logger, MLFlowLogger):
            if image.ndim == 3:
                image = image.permute(1,2,0)
            image = image.detach().numpy()
            self.backend_logger.experiment.log_image(key=key, image=image, run_id=self.backend_logger.run_id)

    @rank_zero_only
    def log_dataframe(self, df: pd.DataFrame, filename):
        if isinstance(self.backend_logger, MLFlowLogger):
            filename = Path(filename)
            filename = filename.with_suffix('.json')
            self.backend_logger.experiment.log_table(data=df, 
                                          artifact_file=filename._str,
                                          run_id=self.backend_logger.run_id)
        elif isinstance(self.backend_logger, TensorBoardLogger):
            filename = Path(filename)
            filename = filename.with_suffix('.csv')
            path_csv = Path(self.log_dir) / filename
            df.to_csv(path_csv, index=False)
            
            
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        if isinstance(self.backend_logger, MLFlowLogger):
            self.backend_logger.finalize(status=status)
        
            
class SegmentationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: ModelABC,
        type_segmentation: Literal["binary", "multiclass", "multilabel", "regression"],
        loss: Callable,
        padding_strategy: Literal['none', 'apply_and_undo'] = 'none'
    ) -> None:
        """A lightning module adapted for segmentation of weather images.

        Args:
            model (ModelABC): Torch neural network model in [DeepLabV3, DeepLabV3Plus, HalfUNet, Segformer, SwinUNETR, UNet, CustomUnet, UNETRPP]
            type_segmentation (Literal["binary", "multiclass", "multilabel", "regression"]): Type of segmentation we want to do"
            loss (Callable): Loss function
            padding_stratey (Literal['none', 'apply_and_undo']): Defines the padding strategy to use. With 'none', it's is up to the user to
                make sure that the input shapes fit the underlying model.
                With 'apply_and_undo', padding is applied for the forward pass, but it is undone before returning the output. 
        """
        super().__init__()
        self.model = model
        self.channels_last = self.model.in_channels == 3
        if self.channels_last:  # Optimizes computation for RGB images
            self.model = self.model.to(memory_format=torch.channels_last)
        self.type_segmentation = type_segmentation
        self.loss = loss
        self.metrics = self.get_metrics()

        # class variables to log metrics for each sample during train/test step
        self.test_metrics = {}
        self.training_loss = []
        self.validation_loss = []

        self.save_hyperparameters(ignore=["loss", "model"])

        # example array to get input / output size in model summary and graph of model:
        self.example_input_array = torch.Tensor(
            8,
            self.model.in_channels,
            self.model.input_shape[0],
            self.model.input_shape[1],
        )
        
        self.padding_strategy = padding_strategy
        if not self.model.auto_padding_supported and padding_strategy != 'none':
            warnings.warn(f"{self.model.__class__.__name__} does not support autopadding and will not be used.",
                          UserWarning)

    def get_metrics(self):
        """Defines the metrics that will be computed during valid and test steps."""

        if self.type_segmentation == "regression":
            metrics_dict = torch.nn.ModuleDict(
                {
                    "rmse": tm.MeanSquaredError(squared=False),
                    "mae": tm.MeanAbsoluteError(),
                    "mape": tm.MeanAbsolutePercentageError(),
                }
            )
        else:
            metrics_kwargs = {"task": self.type_segmentation}
            acc_kwargs = {"task": self.type_segmentation}

            if self.type_segmentation == "multiclass":
                metrics_kwargs["num_classes"] = self.model.out_channels
                acc_kwargs["num_classes"] = self.model.out_channels
                # by default, average="micro" and when task="multiclass", f1 = recall = acc = precision
                # consequently, we put average="macro" for other metrics
                metrics_kwargs["average"] = "macro"
                acc_kwargs["average"] = "micro"

            elif self.type_segmentation == "multilabel":
                metrics_kwargs["num_labels"] = self.model.out_channels
                acc_kwargs["num_labels"] = self.model.out_channels

            metrics_dict = {
                "acc": tm.Accuracy(**acc_kwargs),
                "f1": tm.F1Score(**metrics_kwargs),
                "recall_pod": tm.Recall(**metrics_kwargs),
                "precision": tm.Precision(**metrics_kwargs),  # Precision = 1 - FAR
            }
        return torch.nn.ModuleDict(metrics_dict)

    def forward(self, inputs: torch.Tensor):
        """Runs data through the model. Separate from training step."""
        if self.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        # We prefer when the last activation function is included in the loss and not in the model.
        # Consequently, we need to apply the last activation manually here, to get the real output.
        inputs, old_shape = self._maybe_padding(data_tensor=inputs)
        y_hat = self.model(inputs)
        y_hat = self.last_activation(y_hat)
        y_hat = self._maybe_unpadding(y_hat, old_shape=old_shape)
        return y_hat

    def _shared_forward_step(self, x: torch.Tensor, y: torch.Tensor):
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        # We prefer when the last activation function is included in the loss and not in the model.
        # Consequently, we need to apply the last activation manually here, to get the real output.
        x, old_shape = self._maybe_padding(x)
        y_hat = self.model(x)
        y_hat = self._maybe_unpadding(y_hat, old_shape=old_shape)

        loss = self.loss(y_hat, y)
        y_hat = self.last_activation(y_hat)

        return y_hat, loss
    
    def _maybe_padding(self, data_tensor: torch.Tensor)-> Tuple[torch.Tensor, Optional[torch.Size]]:
        """ Performs an optional padding to ensure that the data tensor can be fed 
            to the underlying model. Padding will happen if the underlying model 
            supports it and if self.padding_strategy is set to 'apply_and_undo'.

        Args:
            data_tensor (torch.Tensor): the input data to be potentially padded. 

        Raises:
            ValueError: if the padding strategy is not valid, an error is raised. 

        Returns:
            Tuple[torch.Tensor, Optional[torch.Size]]: the padded tensor, where the original data is found in the center, 
            and the old size if padding was possible. If not possible or the shape is already fine, 
            the data is returned untouched and the second return value will be none. 
        """
        if self.padding_strategy == 'none' or not self.model.auto_padding_supported:
            return data_tensor, None
        if self.padding_strategy != 'apply_and_undo':
            raise ValueError()
        
        old_shape = data_tensor.shape[-len(self.model.input_shape):]
        valid_shape, new_shape = self.model.validate_input_shape(data_tensor.shape[-len(self.model.input_shape):])
        if not valid_shape:
            return pad_batch(batch=data_tensor, new_shape=new_shape, pad_value=0), old_shape
        return data_tensor, None
    
    def _maybe_unpadding(self, data_tensor: torch.Tensor, old_shape: torch.Size)-> torch.Tensor:
        """Potentially removes the padding previously added to the given tensor. This action 
           is only carried out if self.padding_strategy is set to 'apply_and_undo' and old_shape 
           is not None.

        Args:
            data_tensor (torch.Tensor): The data tensor from which padding is to be removed. 
            old_shape (torch.Size): The previous shape of the data tensor. It can either be 
            [W,H] or [W,H,D] for 2D and 3D data respectively. old_shape is returned by self._maybe_padding.

        Returns:
            torch.Tensor: The data tensor with the padding removed, if possible.
        """
        if self.padding_strategy == 'apply_and_undo' and old_shape is not None:
            return undo_padding(data_tensor, old_shape=old_shape)
        return data_tensor
        

    def on_train_start(self):
        """Setup custom scalars panel on tensorboard and log hparams.
        Useful to easily compare train and valid loss and detect overtfitting."""
        print(f"Logs will be saved in \033[96m{self.logger.log_dir}\033[0m")
        self.logger.add_custom_scalars(layout)
        hparams = dict(self.hparams)
        hparams["loss"] = self.loss.__class__.__name__
        hparams["model"] = self.model.__class__.__name__
        self.logger.log_hyperparams(hparams, {"val_loss": 0, "val_f1": 0})

    def _shared_epoch_end(self, outputs: torch.Tensor, label: torch.Tensor):
        """Computes and logs the averaged loss at the end of an epoch on custom layout.
        Step shared by training and validation epochs.
        """
        avg_loss = torch.stack(outputs).mean()
        lg = self.logger.experiment
        if isinstance(lg, MlflowClient):
            lg.log_metric(run_id=self.logger.run_id, key=f"loss/{label}", value=avg_loss, step=self.current_epoch)
        else:
            lg.add_scalar(f"loss/{label}", avg_loss, self.current_epoch)

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        _, loss = self._shared_forward_step(x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_loss.append(loss)
        return loss

    def on_train_epoch_end(self):
        self._shared_epoch_end(self.training_loss, "train")
        self.training_loss.clear()  # free memory

    def val_plot_step(self, batch_idx: int, y: torch.Tensor, y_hat: torch.Tensor):
        """Plots images on first batch of validation and log them in logger.
        Should be overwrited for each specific project, with matplotlib plots."""
        if batch_idx == 0:
            lg = self.logger.experiment
            step = self.current_epoch
            dformat = "HW" if self.type_segmentation == "multiclass" else "CHW"
            
                
            if step == 0:
                self.logger.log_image(key="val_plots/true_image",
                                      image=y[0],
                                      step=step, dataformats=dformat)
            self.logger.log_image(key='val_plots/pred_image',
                                  image=y[0],
                                  step=step, dataformats=dformat)

    def validation_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.validation_loss.append(loss)
        y_hat = self.probabilities_to_classes(y_hat)
        for metric in self.metrics.values():
            metric.update(y_hat, y)
        self.val_plot_step(batch_idx, y, y_hat)
        return loss

    def on_validation_epoch_end(self):
        self._shared_epoch_end(self.validation_loss, "validation")
        self.validation_loss.clear()  # free memory
        for metric_name, metric in self.metrics.items():
            # Use add scalar to log at step=current_epoch
            lg = self.logger.experiment
            if isinstance(lg, MlflowClient):
                lg.log_metric(run_id=self.logger.run_id, key=f"val_{metric_name}", value=metric.compute(), step=self.current_epoch)
            else:
                lg.add_scalar(f"val_{metric_name}", metric.compute(), self.current_epoch)
            metric.reset()

    def test_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int):
        """Computes metrics for each sample, at the end of the run."""
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        y_hat = self.probabilities_to_classes(y_hat)

        # Save metrics values for each sample
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics.items():
            metric.update(y_hat, y)
            batch_dict[metric_name] = metric.compute()
            metric.reset()
        self.test_metrics[batch_idx] = batch_dict

    def build_metrics_dataframe(self) -> pd.DataFrame:
        data = []
        first_sample = list(self.test_metrics.keys())[0]
        metrics = list(self.test_metrics[first_sample].keys())
        for name_sample, metrics_dict in self.test_metrics.items():
            data.append([name_sample] + [metrics_dict[m].item() for m in metrics])
        return pd.DataFrame(data, columns=["Name"] + metrics)

    # @rank_zero_only
    # def save_test_metrics_as_csv(self, df: pd.DataFrame) -> None:
    #     path_csv = Path(self.logger.log_dir) / "metrics_test_set.csv"
    #     df.to_csv(path_csv, index=False)
    #     print(f"--> Metrics for all samples saved in \033[91m\033[1m{path_csv}\033[0m")

    def on_test_epoch_end(self):
        """Logs metrics in logger hparams view, at the end of run."""
        df = self.build_metrics_dataframe()
        # self.save_test_metrics_as_csv(df)
        self.logger.log_dataframe(df=df, filename="metrics_test_set.csv")
        df = df.drop("Name", axis=1)

    def last_activation(self, y_hat: torch.Tensor):
        """Applies appropriate activation according to task."""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.log_softmax(dim=1).exp()
        elif self.type_segmentation in ["binary", "multilabel"]:
            y_hat = torch.nn.functional.logsigmoid(y_hat).exp()
        return y_hat

    def probabilities_to_classes(self, y_hat: torch.Tensor):
        """Transfrom probalistics predictions to discrete classes"""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.argmax(dim=1)
        elif self.type_segmentation in ["binary", "multilabel"]:
            # Default detection threshold = 0.5
            y_hat = (y_hat > 0.5).int()
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
