import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from dretino.dataloader.build_features import CustomDataset
from dretino.models.coralloss import cal_coral_loss
from dretino.models.cornloss import cal_corn_loss
from dretino.models.crossentropy import ce_loss
from dretino.models.mseloss import mse_loss


def create_preds_data_loader(
    df_train,
    df_valid,
    df_test,
    train_path,
    valid_path,
    test_path,
    train_file_ext,
    val_file_ext,
    test_file_ext,
    transforms,
):
    train_data = CustomDataset(
        df_train, train_path, train_file_ext, transform=transforms
    )

    val_data = CustomDataset(df_valid, valid_path, val_file_ext, transform=transforms)

    test_data = CustomDataset(df_test, test_path, test_file_ext, transform=transforms)

    train_dataloader = DataLoader(
        train_data, batch_size=1, num_workers=2, shuffle=False
    )

    val_dataloader = DataLoader(val_data, batch_size=1, num_workers=2, shuffle=False)

    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=2, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def get_perds(path, Model, dataloader, loss_fn, num_classes):
    """
    Function to get preds

    Parameters
    ----------
    path : str, ckpt filepath
    Model : LightningModule,
    dataloader : train/val/test dataloader,
    loss_fn : str, loss function used
    num_classes = int, number of classes
    """
    model = Model.load_from_checkpoint(path + ".ckpt")
    cmat = ConfusionMatrix(num_classes=num_classes)
    preds_list = []
    y_list = []
    logits_list = []
    model.eval()
    for x, y in tqdm(dataloader):
        with torch.no_grad():
            logits = model(x)
        if loss_fn == "ce":
            loss, preds, y = ce_loss(logits, y)
        elif loss_fn == "mse":
            loss, preds, y = mse_loss(logits, y)
        elif loss_fn == "corn":
            loss, preds, y = cal_corn_loss(logits, y, num_classes)
        elif loss_fn == "coral":
            loss, preds, y = cal_coral_loss(logits, y, num_classes)
        else:
            s = (
                'Invalid value for `loss`. Should be "ce",\
                 "mse", "corn" or "coral". Got %s'
                % loss_fn
            )
            raise ValueError(s)
        cmat(preds, y)
        preds_list.append(preds.numpy())
        y_list.append(y.numpy())
        logits_list.append(logits.numpy())
        # np.squeeze(torch.nn.Softmax(dim=-1)(torch.tensor(logits_list)).numpy(),

    return (
        np.squeeze(logits_list, axis=1),
        np.array(y_list).ravel(),
        cmat.compute().numpy(),
    )
