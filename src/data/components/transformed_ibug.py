import albumentations as A
from typing import Optional 
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.ibug import Ibug

class TransformedIbug(Dataset):
    def __init__(self, Dataset: Dataset = Ibug, transform: Optional[A.Compose] = None) -> None:
        self.pre_data = Dataset
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()],A.KeypointParams(format='xy', remove_invisible=False))
    
    def __len__ (self) -> int:
        return len(self.pre_data)
        
    def __getitem__(self, index: int):
        transformed_data = self.transform(image = np.array(self.pre_data[index]['image']), keypoints = self.pre_data[index]['keypoints'])
        image = transformed_data['image']
        keypoints = (np.array(transformed_data['keypoints']) / image.shape[1:]).astype(np.float32)
        return image, keypoints