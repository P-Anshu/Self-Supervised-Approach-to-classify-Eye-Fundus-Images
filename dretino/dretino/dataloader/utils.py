import os

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, dfx, image_dir, file_ext, transform=None):
        """Create a pytorch dataset

        Parameters
        ----------
        dfx : (DataFrame), DataFrame containing image name and retinopathy grade
        image_dir : (str), path of the image directory
        transform : (Albumentations, optional), Transformations. Defaults to None.
        """
        super().__init__()
        self.dfx = dfx
        self.image_ids = self.dfx.iloc[:, 0].values
        self.targets = self.dfx.iloc[:, 1].values
        self.num_classes = self.dfx.iloc[:, 1].nunique()
        self.image_dir = image_dir
        self.transform = transform
        self.file_ext = file_ext

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        index = torch.tensor(self.targets[idx])
        target = F.one_hot(index, num_classes=self.num_classes)

        img = np.array(
            Image.open(os.path.join(self.image_dir, img_name + self.file_ext)).convert(
                "RGB"
            )
        )
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, target


def get_sampler(train_data):
    """Create a Sampler to Balance and imbalanced dataset while loading

    Parameters
    ----------
    train_data : (DataLoader), Train DataLoader to get the sampler

    Returns
    -------
    WeightedRandomSampler : Sampler
    """
    targets = []
    for _, target in tqdm(train_data):
        targets.append(torch.argmax(target, dim=-1))

    targets = torch.stack(targets)
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
    )
    weight = 1.0 / class_sample_count.float()
    sample_weight = torch.tensor([weight[t] for t in targets])
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

    return sampler
