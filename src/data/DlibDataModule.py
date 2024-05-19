from lightning import LightningDataModule
import albumentations as transfrom
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.Dlib import Dlib
import torch
import pyrootutils
from src.data.components.TransformedDlib import TransformedDlib

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


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

            data_train, data_val, data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42))

            self.data_train = TransformedDlib(data_train, self.train_transform)
            self.data_val = TransformedDlib(data_val, self.val_transform)
            self.data_test = TransformedDlib(data_test, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False)
    
    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(device)
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    @staticmethod
    def draw_batch(batch):
        