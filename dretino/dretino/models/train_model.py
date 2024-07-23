import os
from datetime import datetime

import pytorch_lightning as pl

from dretino.models.coralloss import ModelCORAL, cal_coral_loss
from dretino.models.cornloss import ModelCORN, cal_corn_loss
from dretino.models.crossentropy import ModelCE, ce_loss
from dretino.models.mseloss import ModelMSE, mse_loss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, CohenKappa


class Model(pl.LightningModule):
    def __init__(
        self,
        loss="ce",
        model_name="resnet50",
        additional_layers=True,
        num_classes=5,
        lr=3e-4,
        num_neurons=512,
        n_layers=2,
        dropout_rate=0.2,
        max_epochs=40,
    ):
        """Pytorch Lightning Module for Model

        Parameters
        ----------
        loss : (str, optional): Loss function to use ['ce','mse','corn','coral']. Defaults to 'ce'.
        model_name : (str, optional): Model name for timm. Defaults to 'resnet50d'.
        additional_layers : (bool, optional) Add addtional layers Defaults to True
        num_classes : (int, optional): Number of classes. Defaults to 5.
        lr : (_type_, optional): Learning Rate. Defaults to 3e-4.
        num_neurons : (int, optional): Num of Neurons in the first layer. Defaults to 512.
        n_layers : (int, optional): Number of additional layers. Defaults to 2.
        dropout_rate : (float, optional): Dropout Rate Defaults to 0.2.

        Raises
        ------
        ValueError : Should be "ce", "mse", "corn" or "coral"
        """
        super(Model, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.loss = loss
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.additional_layers = additional_layers
        self.lr = lr
        self.model_name = model_name
        self.n_layers = n_layers
        self.num_neurons = num_neurons
        self.dropout_rate = dropout_rate
        self.accuracy = Accuracy()
        self.metric = F1Score(num_classes=self.num_classes, average="macro")
        self.kappametric = CohenKappa(num_classes=self.num_classes)
        if self.loss == "ce":
            self.model = ModelCE(
                self.model_name,
                self.num_classes,
                self.additional_layers,
                self.num_neurons,
                self.n_layers,
                self.dropout_rate,
            )
        elif self.loss == "mse":
            self.model = ModelMSE(
                self.model_name,
                self.num_classes,
                self.additional_layers,
                self.num_neurons,
                self.n_layers,
                self.dropout_rate,
            )
        elif self.loss == "corn":
            self.model = ModelCORN(
                self.model_name,
                self.num_classes,
                self.additional_layers,
                self.num_neurons,
                self.n_layers,
                self.dropout_rate,
            )
        elif self.loss == "coral":
            self.model = ModelCORAL(
                self.model_name,
                self.num_classes,
                self.additional_layers,
                self.num_neurons,
                self.n_layers,
                self.dropout_rate,
            )
        else:
            s = (
                'Invalid value for `loss`. Should be "ce", '
                '"mse", "corn" or "coral". Got %s' % self.loss
            )
            raise ValueError(s)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds, loss, acc, f1_score, kappa_score = self._get_preds_loss_accuracy(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.log("train_kappa", kappa_score, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc, f1_score, kappa_score = self._get_preds_loss_accuracy(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_kappa", kappa_score, prog_bar=False, on_epoch=True)
        self.log("valid_F1_score", f1_score, prog_bar=False, on_epoch=True)
        return preds

    def test_step(self, batch, batch_idx):
        _, loss, acc, f1_score, kappa_score = self._get_preds_loss_accuracy(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        self.log("test_F1_score", f1_score, prog_bar=True)
        self.log("test_kappa", kappa_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )

        return [optimizer], [scheduler]

    def _get_preds_loss_accuracy(self, batch):
        x, y = batch
        logits = self(x)
        if self.loss == "ce":
            loss, preds, y = ce_loss(logits, y)
        elif self.loss == "mse":
            loss, preds, y = mse_loss(logits, y)
        elif self.loss == "corn":
            loss, preds, y = cal_corn_loss(logits, y, self.num_classes)
        elif self.loss == "coral":
            loss, preds, y = cal_coral_loss(logits, y, self.num_classes)
        else:
            s = (
                'Invalid value for `loss`. Should be "ce", '
                '"mse", "corn" or "coral". Got %s' % self.loss
            )
            raise ValueError(s)

        acc = self.accuracy(preds, y)
        f1_score = self.metric(preds, y)
        kappa_score = self.kappametric(preds, y)
        return preds, loss, acc, f1_score, kappa_score


def train(Model, dm, wab=False, fast_dev_run=False, overfit_batches=False, **kwargs):
    """Create a trainer to save tensorboard/wandb/csv_logger checkpoints.

    Parameters
    ----------
    Model : (LightningModule), Model
    dm : (LightningDataModule), DataModule
    wab : (bool, optional), Wandb integration. Defaults to False.
    fast_dev_run : (bool, optional), Fast dev run for unit test of the model code. Defaults to False.
    overfit_batches : (bool, optional), Used to overfit batches for sanity check. Defaults to False.

    Returns
    -------
    file_name : _str_, Filename created using datetime
    trainer : _LightningTrainer_, Trainer used in training
    """
    if wab and fast_dev_run:
        s = "Both wab and fast_dev_run cannot be true at the same time"
        raise RuntimeError(s)
    if fast_dev_run and overfit_batches:
        s = "Both overfit_batches and fast_dev_run cannot be true at the same time"
        raise RuntimeError(s)
    num_neurons = kwargs["num_neurons"]
    num_layers = kwargs["num_layers"]
    dropout = kwargs["dropout"]
    lr = kwargs["lr"]
    loss = kwargs["loss"]
    model_name = kwargs["model_name"]
    additional_layers = kwargs["additional_layers"]
    save_dir = kwargs["save_dir"]

    model = Model(
        model_name=model_name,
        loss=loss,
        additional_layers=additional_layers,
        num_neurons=num_neurons,
        n_layers=num_layers,
        dropout_rate=dropout,
        lr=lr,
        num_classes=5,
        max_epochs=kwargs["epochs"],
    )

    file_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{loss}_{num_neurons}_{num_layers}_{dropout}_{lr}"
    csv_logger = CSVLogger(save_dir=save_dir, name=file_name)
    tensorboard_logger = TensorBoardLogger(save_dir=save_dir, name=file_name)
    if wab:
        wandb_logger = WandbLogger(project=kwargs["project"], log_model=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=save_dir,
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        filename=file_name,
        verbose=True,
        mode="max",
    )
    trainer = Trainer(
        gpus=kwargs["gpus"],
        strategy=kwargs["strategy"],
        tpu_cores=kwargs["tpus"],
        max_epochs=kwargs["epochs"],
        callbacks=[checkpoint_callback],
        logger=[csv_logger, tensorboard_logger],
    )
    if wab:
        wandb_logger.watch(model)
        trainer = Trainer(
            gpus=kwargs["gpus"],
            strategy=kwargs["strategy"],
            tpu_cores=kwargs["tpus"],
            max_epochs=kwargs["epochs"],
            callbacks=[checkpoint_callback],
            logger=[csv_logger, tensorboard_logger, wandb_logger],
        )
    if fast_dev_run:
        trainer = Trainer(
            gpus=kwargs["gpus"],
            strategy=kwargs["strategy"],
            tpu_cores=kwargs["tpus"],
            fast_dev_run=fast_dev_run,
        )
    if overfit_batches:
        trainer = Trainer(
            gpus=kwargs["gpus"],
            strategy=kwargs["strategy"],
            tpu_cores=kwargs["tpus"],
            overfit_batches=1,
            max_epochs=kwargs["epochs"],
            callbacks=[checkpoint_callback],
            logger=[csv_logger, tensorboard_logger],
        )
    if wab and overfit_batches:
        trainer = Trainer(
            gpus=kwargs["gpus"],
            strategy=kwargs["strategy"],
            tpu_cores=kwargs["tpus"],
            max_epochs=kwargs["epochs"],
            callbacks=[checkpoint_callback],
            logger=[csv_logger, tensorboard_logger, wandb_logger],
        )

    trainer.fit(model, dm)
    return os.path.join(save_dir, file_name), trainer
