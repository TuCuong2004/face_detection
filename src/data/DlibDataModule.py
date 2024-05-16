from lightning import LightningDataModule
import albumentations as transfrom
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.Dlib import Dlib

class DlibDataModule(LightningDataModule):
    def __init__(
            self,
            train_transform: Optional[transfrom.Compose] = None,
            val_transform: Optional[transfrom.Compose] = None,
            data_dir: str = "data/IBUG",
            train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
            batch_size: int = 32,
            num_workers: int = 8,
            pin_memory: bool = False,
            ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10        
    
    def setup(self, stage: Optional[str] = None):

        if not self.data_train and not self.data_val and not self.data_test:
            
            dataset = Dlib()