import copy

import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from lightly.data import LightlyDataset
from lightly.data import DINOCollateFunction
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        input_dim = 2048

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        update_momentum(self.student_backbone, self.teacher_backbone, m=0.99)
        update_momentum(self.student_head, self.teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


model = DINO()

dataset = LightlyDataset(input_dir="../aptos/train_images_resize/")

collate_fn = DINOCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

gpus = [0, 1, 2, 3]

wandb_logger = WandbLogger(project="ssl_aptos")

trainer = pl.Trainer(
    max_epochs=250,
    gpus=gpus,
    strategy="ddp",
    sync_batchnorm=True,
    replace_sampler_ddp=True,
    logger=wandb_logger,
)
trainer.fit(model=model, train_dataloaders=dataloader)
