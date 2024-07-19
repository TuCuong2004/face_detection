import albumentations as A
import typing as typ
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class TransformedIbug(Dataset):
    def __init__(self, pre_data: Dataset, transform: typ[A.Compose] = None) -> None:
        if transform:
            self.transform = transform
        else:
