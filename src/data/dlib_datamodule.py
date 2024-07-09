from lightning import LightningDataModule
import albumentations as transfrom
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch
import pyrootutils
from matplotlib import pyplot as plt
import hydra
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.transformed_dlib import TransformedDlib
from src.data.components.dlib import Dlib


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
        self.batch_size = batch_size
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10        

    def prepare_data(self):
        pass

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
    def drawBatch(batch, width=8, height=4):

        mean = [0.485, 0.456, 0.406]   # Assuming RGB images
        std = [0.229, 0.224, 0.225]

        images, keypoints = batch
       
        images = images.numpy().transpose(0, 2, 3, 1)
        images = images * std + mean

        fig, axs = plt.subplots(4, 8, figsize=(30, 10))
        for i in range(height):
            for j in range(width):
                idx = i*8 + j   
                axs[i][j].imshow(images[idx])
                axs[i][j].scatter(keypoints[idx][:, 0]*224, keypoints[idx][:, 1]*224, s=1, c='r')
                axs[i][j].axis('off')
        
        plt.show()

        plt.savefig('batchDrawers.png')


@hydra.main(version_base=None, config_path="../../configs/data", config_name="dlib.yaml")
def main(cfg):
    dm = hydra.utils.instantiate(cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    DlibDataModule.drawBatch(batch=batch)


if __name__ == '__main__':
    main()
