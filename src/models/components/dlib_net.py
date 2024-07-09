import hydra
from omegaconf import DictConfig
import pyrootutils
from torchvision import models
import torch

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dlib_datamodule import DlibDataModule

class DlibNet(torch.nn.Module):
    def __init__(self, 
                 name : str, 
                 output_shape : tuple =(68,2)):
        super(DlibNet, self).__init__()
        self.output_shape = output_shape
        self.model = models.get_model(name, num_classes = output_shape[0] * output_shape[1])
    

    def forward(self, x):
        af = self.model(x)
        af = torch.reshape(af, [af.shape[0], self.output_shape[0], self.output_shape[1]])
        return af


def main():
    model = DlibNet("resnet18", (68,2))
    pred = model(torch.rand([1, 3, 224, 224]))
    print(pred.dtype)


if __name__ == "__main__":
    _ = main()

