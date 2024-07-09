from lightning import LightningModule
import torch
from torchmetrics.regression.mae import MeanAbsoluteError as MAE
from torchmetrics import MaxMetric, MeanMetric
from typing import Any
import pyrootutils
from lightning.pytorch.trainer import Trainer
import hydra

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class DlibModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.criterion = torch.nn.MSELoss()
        self.net = net

        self.train_acc = MAE()
        self.val_acc = MAE()
        self.test_acc = MAE()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y
    
    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss, preds, y = self.model_step(batch=batch)
        loss = self.train_loss(loss)
        self.train_acc(preds, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, preds, y = self.model_step(batch=batch)
        loss = self.val_loss(loss)
        self.val_acc(preds, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, preds, y = self.model_step(batch=batch)
        loss = self.test_loss(loss)
        self.test_acc(preds, y)
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
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

@hydra.main(config_path="../../configs/model", config_name="dlib")
def main(cfg):
    model = hydra.utils.instantiate(cfg)
    print(model)

if __name__ == "__main__":
    main()