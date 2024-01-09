from typing import Any

import lightning.pytorch as pl
import omegaconf
import torch
import torch.nn as nn


def conv_block_3x3(in_channels: int, out_channels: int, stride: int = 1) -> torch.Tensor:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
    )


class Flatten(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(tensor, start_dim=1)


class MyModel(pl.LightningModule):
    def __init__(self, conf: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.model = nn.Sequential(
            conv_block_3x3(conf["model"]["in_feature"], conf["model"]["in_feature"] * 10),
            nn.MaxPool2d(2),
            conv_block_3x3(
                conf["model"]["in_feature"] * 10, conf["model"]["in_feature"] * 100
            ),
            nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )
        self.dropout = nn.Dropout(p=conf["model"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(
                conf["model"]["in_feature"] * 100,
                conf["model"]["embedding_size"],
                bias=False,
            ),
            nn.BatchNorm1d(num_features=conf["model"]["embedding_size"]),
            nn.ReLU(),
        )
        self.pred = nn.Sequential(
            nn.Linear(
                conf["model"]["embedding_size"], conf["data"]["num_classes"], bias=False
            )
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.model(tensor)
        tensor = self.dropout(tensor)
        tensor = self.fc(tensor)
        tensor = self.dropout(tensor)
        tensor = self.pred(tensor)
        return tensor

    def training_step(self, batch: Any, batch_idx: int):
        """Compute and return the training loss and some additional metrics
        for e.g. the progress bar or logger.

        Args:
            batch: The output of DataLoader
                A tensor, tuple or list.
            batch_idx (``int``): Integer displaying index of this batch

        Return:
            Any of.

            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the
                key ``'loss'``
            - ``None`` - Training will skip to the next batch. This is only for automatic
                optimization. This is not supported for multi-GPU, TPU, IPU, or
                    DeepSpeed.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        """
        X_batch, y_batch = batch
        y_preds = self(X_batch)
        loss = self.loss_fn(y_preds, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        """Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like
        accuracy.

        Return:
            - Any object or value
            - ``None`` - Validation will skip to the next batch

        Examples::

            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})

        """
        X_batch, y_batch = batch
        y_preds = self(X_batch)
        loss = self.loss_fn(y_preds, y_batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        """Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest
        such as accuracy.

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.

        Return:
           Any of.

            - Any object or value
            - ``None`` - Testing will skip to the next batch

        Examples::

            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'test_loss': loss, 'test_acc': test_acc})

        """
        X_batch, y_batch = batch
        logits = self(X_batch)
        acc = self.accuracy(logits, y_batch)
        self.log("test_acc", acc)

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        """By default, it calls forward method.
        Override to add any processing logic.

        It is used to scale inference on multi-devices.

        To prevent an OOM error, it is possible to use
        :class:`~pytorch_lightning.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each batch or on
        epoch end.

        The :class:`~pytorch_lightning.callbacks.BasePredictionWriter` should be used
        while using a spawn based accelerator.
        This happens for ``Trainer(strategy="ddp_spawn")``
        or training on 8 TPU cores with ``Trainer(accelerator="tpu", devices=8)``
        as predictions won't be returned.

        Example ::

            class MyModel(LightningModule):

                def predict_step(self, batch, batch_idx, dataloader_idx=0):
                    return self(batch)

            dm = ...
            model = MyModel()
            trainer = Trainer(accelerator="gpu", devices=2)
            predictions = trainer.predict(model, dm)

        """
        X_batch, _ = batch
        return self(X_batch)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.conf["train"]["learning_rate"])
