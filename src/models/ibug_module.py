from typing import Any
from lightning import LightningModule
import torch
from torchmetrics.regression.mae import MeanAbsoluteError as MAE
from torchmetrics import MaxMetric, MeanMetric
import hydra
import pyrootutils
import numpy as np
import copy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import src.models.spiga.inference.pretreatment as pretreat
from src.data.components.ibug_config import AlignConfig
from src.data.components.ibug import get_dataset

class IbugModule(LightningModule):
    def __init__(self,
                net: torch.nn.modules,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.MSELoss()

        self.train_acc = MAE()
        self.val_acc = MAE()
        self.test_acc = MAE()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        config = AlignConfig(database_name="wflw", mode="train")
        config.update({"generate_pose": False, "heatmap2D_norm": False})
        dataset = get_dataset(config, debug=True)


        loader_3DM = pretreat.AddModel3D(ldm_ids = dataset[1]['ids_ldm'],
                                         totensor=True)
        # params_3DM = self._data2device(loader_3DM())
        self.model3d = loader_3DM['model3d']
        self.cam_matrix = loader_3DM['cam_matrix']
        
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.tensor):
        return self.net(x)
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
    
    # def model_step(self, batch):
    #     x, y = batch
    #     predict = self.forward(x)
    #     loss = self.criterion(predict, y)
    #     return loss 

    def pretreat(self, image, bboxes):
        crop_bboxes = []
        crop_images = []
        for bbox in bboxes:
            sample = {'image': copy.deepcopy(image),
                      'bbox': copy.deepcopy(bbox)}
            sample_crop = self.transforms(sample)
            crop_bboxes.append(sample_crop['bbox'])
            crop_images.append(sample_crop['image'])

        # Images to tensor and device
        batch_images = torch.tensor(np.array(crop_images), dtype=torch.float)
        batch_images = self._data2device(batch_images)
        # Batch 3D model and camera intrinsic matrix
        batch_model3D = self.model3d.unsqueeze(0).repeat(len(bboxes), 1, 1)
        batch_cam_matrix = self.cam_matrix.unsqueeze(0).repeat(len(bboxes), 1, 1)

        model_inputs = [batch_images, batch_model3D, batch_cam_matrix]
        return model_inputs, crop_bboxes

    def training_step(self, batch):
        
        pretreat(batch['image'], batch['bbox'])
        predict = self.forward(batch)
        # loss = self.train_loss(self.criterion(y, predict))
        # self.train_acc(predict, y)
        return predict
    
    def validation_step(self, batch):
        # x, y = batch
        pretreat(batch['image'], batch['bbox'])
        predict = self.forward(batch)
        # loss = self.val_loss(self.criterion(y, predict))
        # self.val_acc(predict, y)
        return predict
    
    def test_step(self, batch):
        # x, y = batch
        pretreat(batch['image'], batch['bbox'])
        predict = self.forward(batch)
        # loss = self.test_loss(self.criterion(y, predict))
        # self.test_acc(predict, y)
        return predict
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        pass
    
    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) :
            """Choose what optimizers and learning-rate schedulers to use in your optimization.
            Normally you'd need one. But in the case of GANs or similar you might have multiple.

            Examples:
                https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
            """
            optimizer = self.hparams.optimizer(params=self.parameters())
            if self.hparams.scheduler is not None:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}
    
@hydra.main(version_base=None, config_path='../../configs/model', config_name='dlib')
def main(cfg):
    model = hydra.utils.instantiate(cfg)
    print(model)

if __name__ == '__main__':
    main()



