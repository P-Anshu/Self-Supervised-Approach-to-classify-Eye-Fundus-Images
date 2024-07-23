import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes


class SwaV(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SwaVProjectionHead(2048, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)

        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


wandb_logger = WandbLogger(project="ssl_aptos")

model = SwaV()

dataset = LightlyDataset(input_dir="../aptos/train_images_resize/")

collate_fn = SwaVCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

gpus = [4, 5, 6, 7]

trainer = pl.Trainer(
    max_epochs=150, gpus=gpus, strategy="ddp", sync_batchnorm=True, logger=wandb_logger
)
trainer.fit(model=model, train_dataloaders=dataloader)
