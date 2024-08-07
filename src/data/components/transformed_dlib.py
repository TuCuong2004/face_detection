from torch.utils.data import Dataset
from typing import Optional
import albumentations as A
import pyrootutils
from albumentations.pytorch import ToTensorV2
import numpy as np


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.dlib import Dlib


class TransformedDlib(Dataset):
    def __init__(self, pre_data: Dlib, transform):
        self.pre_data = pre_data
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __len__(self):
        return len(self.pre_data)

    def __getitem__(self, index: int):
        transformed_data = self.transform(image=np.array(self.pre_data[index]['image']),keypoints=self.pre_data[index]['keypoints'])
        image = transformed_data['image']
        keypoints = (np.array(transformed_data['keypoints']) / image.shape[1:]).astype(np.float32)

        return image, keypoints


if __name__ == '__main__':
    batch = TransformedDlib(Dlib(), None)
    image, keypoints = batch[0]
    print(keypoints.shape)
    print(1)
