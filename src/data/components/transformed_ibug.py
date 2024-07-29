import albumentations as A
import typing as typ
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import pyrootutils
from ibug import Ibug
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class TransformedIbug(Dataset):
    def __init__(self, pre_data: Dataset = Ibug, transform: typ[A.Compose] = None) -> None:
        self.pre_data = pre_data
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.resize(224, 224),
                A.normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()],A.KeypointParams(format='xy', remove_invisible=False))
    
    def __len__ (self) -> int:
        return len(self.pre_data)
        
    def __getitem__(self, index):
        transformed_data = self.transform(image = self.pre_data[index]['image'], keypoints = self.pre_data[index]['keypoints'])
        image = transformed_data['image']
        keypoints = (np.array(transformed_data['keypoints']) / image.shape[1:]).astype(np.float32)
        return image, keypoints