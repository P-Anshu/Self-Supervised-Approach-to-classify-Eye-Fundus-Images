import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from dretino.dataloader.utils import get_sampler, CustomDataset
from torch.utils.data import DataLoader


def create_dataloader(
    df_train,
    df_valid,
    df_test,
    train_path,
    valid_path,
    test_path,
    train_file_ext,
    val_file_ext,
    test_file_ext,
    train_transforms,
    val_transforms,
    test_transforms,
):
    """Create Dataloaders

    Parameters
    ----------
    df_train : (DataFrame), Train dataframe
    df_valid : (DataFrame), Val dataframe
    df_test : (DataFrame), Test dataframe
    train_path : (str), Train path
    valid_path : (str), Val path
    test_path : (str), Test path
    train_transforms : (Albumentation), Train transformations
    val_transforms : (Albumentation), Val transformations
    test_transforms : (Albumentation), Test transformations

    Returns
    -------
    train_data : Train Dataset
    val_data : Val Dataset
    test_data : Test Dataset
    sampler : WeightedSampler
    """
    train_data = CustomDataset(
        df_train,
        train_path,
        train_file_ext,
        transform=A.Compose([A.Resize(width=1, height=1), ToTensorV2()]),
    )

    val_data = CustomDataset(
        df_valid, valid_path, val_file_ext, transform=val_transforms
    )

    test_data = CustomDataset(
        df_test, test_path, test_file_ext, transform=test_transforms
    )

    sampler = get_sampler(train_data)

    return train_data, val_data, test_data, sampler


class DRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_train,
        df_valid,
        df_test,
        train_path,
        valid_path,
        test_path,
        train_file_ext,
        val_file_ext,
        test_file_ext,
        train_transforms,
        val_transforms,
        test_transforms,
        num_workers=2,
        batch_size=32,
    ):
        """Pytorch Lightning Data Module

        Parameters
        ----------
        df_train : (DataFrame), Train dataframe
        df_valid : (DataFrame), Val dataframe
        df_test  : (DataFrame), Test dataframe
        train_path : (str), Train path
        valid_path : (str), Val path
        test_path : (str), Test path
        train_transforms : (Albumentation), Train transformations
        val_transforms : (Albumentation), Val transformations
        test_transforms : (Albumentation), Test transformations
        num_workers : (int, optional), Num workers. Defaults to 2.
        batch_size : (int, optional), Batch Size. Defaults to 32.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        (
            self.train_data,
            self.val_data,
            self.test_data,
            self.sampler,
        ) = create_dataloader(
            df_train,
            df_valid,
            df_test,
            train_path,
            valid_path,
            test_path,
            train_file_ext,
            val_file_ext,
            test_file_ext,
            train_transforms,
            val_transforms,
            test_transforms,
        )

        self.train_data = CustomDataset(
            df_train, train_path, train_file_ext, transform=self.train_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
