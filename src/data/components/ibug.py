import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from PIL import Image
import tarfile
import requests
import tqdm
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class Ibug(Dataset):
    def __init__(self, root_dir=r'data/ibug/ibug_300w_large_face_landmark_dataset', data_dir = r'data\FilerData.taz.gz'):
        
        self.root_dir = root_dir
        self.data_dir = data_dir
        if not os.path.exists(self.root_dir):
            print("data don't exist")
            os.makedirs(self.root_dir)
            unzip_data(self.data_dir, root_dir = self.root_dir)
        else:
            self.root = ET.parse(os.path.join(self.root_dir, r'labels_ibug_300W.xml')).getroot()
        
    def __len__(self):
        return len(self.root[2])

    def __getitem__(self, index: int):
        box = self.root[2][index]

        image = Image.open(os.path.join(self.root_dir, self.root[2][index].attrib['file'])).convert("RGB")

        box_area = [box[0].attrib['left'], box[0].attrib['top'], 
                        float(box[0].attrib['width']) + float(box[0].attrib['left']), 
                        float(box[0].attrib['top']) + float(box[0].attrib['height'])]
        
        box_area = np.array(box_area,dtype=float)

        image = image.crop(box=box_area)
        # image = np.array(image, dtype=float)
        keypoints = []

        for kp in box[0] :
            keypoints.append([kp.attrib.get('x'),kp.attrib.get('y')])
        
        keypoints = keypoints[0:]
        
        keypoints = np.array(keypoints,dtype=float)
        
        keypoints[:, 0] = keypoints[:, 0] - box_area[0]
        keypoints[:, 1] = keypoints[:, 1] - box_area[1]
        
        return {'image' : image, 'keypoints' : keypoints}


    @staticmethod
    def show_keypoints(image, keypoints):
        plt.imshow(image.premute(1, 2, 0))
        plt.scatter(keypoints[:0]*224, keypoints[:, 1]*224, marker='.', c='r')
        plt.savefig('landmarkdrawers.png')

    @staticmethod
    def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
        ten = x.clone().premute(1, 2, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)

        print(f"Max: {torch.max(t)}, Min: {torch.min(t)}")
        # B, 3, H, W
        return ten

    @staticmethod
    def show_boundingboxes(image, keypoints, box):

        plt.scatter(keypoints[:, 0]*224, keypoints[:, 1]*224,
                    marker=".", c='r')

        box_w, box_h = 224

        ax = plt.gca()

        rect = Rectangle((box[0], box[1]), box_w, box_h, linewidth=1,
                         edgecolor='r', facecolor='none')

        ax.add_patch(rect)

        plt.imshow(denormalize(image))
        plt.savefig('bbdrawers.png')


def unzip_data(data_dir, root_dir):
    with tarfile.open('data\\FilerData.taz.gz', "r:gz") as tar:
        tar.extractall(root_dir)

def download_data():
    url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"
    respond = requests.get(url, stream= True)
    with open(r'data/FilerData.taz.gz', mode= 'wb') as file:
        for chunk in tqdm(respond.iter_content(chunk_size=1024)):
            if chunk:
                file.write(chunk)
                file.flush()

    
        
            
if __name__ == "__main__":
    cg = Ibug()
    print(cg[0]['keypoints'])
