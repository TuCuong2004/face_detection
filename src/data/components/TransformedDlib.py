from torch.utils.data import Dataset
from src.data.components import Dlib
from typing import Optional
import albumentations as A
import pyrootutils
from albumentations.pytorch import ToTensorV2
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class TransformedfDlib(Dataset):
    def __init__(self, pre_data=Dlib, transform: Optional[A.Compose] = None):
        self.pre_data = pre_data
        if transform:
            self.transform = transform
        else:
            transform = A.Compose([
                A.resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.pre_data)

    def __getitem__(self, index):
        transformed_data = self.transform(np.array(self.pre_data[index]['image']), self.pre_data[index]['keypoints'])
        image = transformed_data['image']
        keypoints = (np.array(transformed_data['keypoints']) / image.shape[1:]).astype(np.float32)
    
        return image, keypoints


if __name__ == 'main':
    print(Dlib)
