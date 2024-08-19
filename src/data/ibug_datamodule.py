from lightning import LightningDataModule 
import pyrootutils
from typing import Any, Dict, Optional, Tuple
import albumentations as A
from torch.utils.data import Dataset, random_split, DataLoader
import torch
from matplotlib import pyplot as plt
import hydra

pyrootutils.setup_root(__file__, indicator=".project-root",pythonpath=True)
from src.data.components.ibug_config import AlignConfig
from src.data.components.ibug import get_dataset
from src.data.components.ibug import Ibug


class IbugDataModule(LightningDataModule):
    def __init__(
            self,
            train_transform: Optional[A.Compose] = None,
            val_transform: Optional[A.Compose] = None,
            data_dir: str = "data/IBUG",
            train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
            batch_size: int = 32,
            num_workers: int = 8,
            pin_memory: bool = False,
            ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        # self.train_val_test_split = train_val_test_split
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

    def setup(self, stage: Optional[str] = None) -> None:

        config = AlignConfig(database_name="wflw", mode="train")
        config.update({"generate_pose": False, "heatmap2D_norm": False})
        dataset = get_dataset(config, debug=True)
        self.pre_data = dataset

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset = self.pre_data,
                lengths = self.hparams.train_val_test_split,
                generator = torch.Generator().manual_seed(42)
            )

        self.data_train = self.data_train.dataset
        self.data_val = self.data_val.dataset
        self.data_test = self.data_test.dataset
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            batch_size = self.batch_size,
            num_workers = self.hparams.num_workers,
            dataset = self.data_train,
            pin_memory = self.hparams.pin_memory,
            shuffle = True
        )

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

        # images, keypoints = batch

        keypoints = batch['mask_ldm']
        images = batch['image']
        print(keypoints.shape)
        images = images.numpy()
        # images = images * std + mean

        fig, axs = plt.subplots(4, 8, figsize=(30, 10))
        for i in range(height):
            for j in range(width):
                idx = i*8 + j   
                axs[i][j].imshow(images[idx])
                # axs[i][j].scatter(keypoints[idx][:, 0]*224, keypoints[idx][:, 1]*224, s=1, c='r')
                axs[i][j].axis('off')

        plt.show()

        plt.savefig('batchDrawers.png')    
    

@hydra.main(version_base=None, config_path='../../configs/data', config_name= 'ibug.yaml')
def main(cfg):
    dm = hydra.utils.instantiate(cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    IbugDataModule.drawBatch(batch=batch)
  
    
if __name__ == '__main__':
    main()