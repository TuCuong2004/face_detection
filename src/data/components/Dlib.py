import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
import tarfile
import torch
import os
from matplotlib.patches import Rectangle


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


class Dlib(Dataset):
    def __init__(self,
                 root_dir=r'data/ibug_300W_large_face_landmark_dataset'):

        xml_path = \
            r'data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W.xml'
        if not os.path.exists(xml_path):
            download_data()
            unzip_data()
        else:
            tree = ET.parse(xml_path)
            self.root = tree.getroot()
            
            self.root_dir = root_dir

    def __len__(self):
        return len(self.root[2])

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.tolist()

        image_name = os.path.join(self.root_dir,
                                self.root[2][index].attrib['file'])

        image = Image.open(image_name).convert("RGB")

        keypoints = []

        for kp in self.root[2][index].iter():
            keypoints.append([kp.attrib.get("x"), kp.attrib.get("y")])

        keypoints = keypoints[2:]

        keypoints = np.array(keypoints, dtype=float)

        box_dict = self.root[2][index][0].attrib

        box = [box_dict.get('left'), box_dict.get('top'),
               float(box_dict.get('left'))+float(box_dict.get('width')),
               float(box_dict.get('top')) + float(box_dict.get('height'))]

        box = np.array(box, dtype=float)

        #
        keypoints[:, 0] = keypoints[:, 0] - float(box_dict.get('left'))
        keypoints[:, 1] = keypoints[:, 1] - float(box_dict.get('top')) 
        # do sth

        image = image.crop(box=box)

        sample = {'image': image, 'keypoints': keypoints}
        return sample

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

    @staticmethod
    def testImage(image):
        t = image.clone().permute(1, 2, 0)
        print(f"Max: {torch.max(t)}, Min: {torch.min(t)}")


def download_data():

    url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"
    response = requests.get(url, stream=True)
    with open(r"data/FilerData.taz.gz", mode="wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                file.write(chunk)
                file.flush()


def unzip_data():

    with tarfile.open('data\\FilerData.taz.gz', "r:gz") as tar:
        tar.extractall('data\\FilerData.taz.gz/')


if __name__ == "__main__":
    print(1)
    dlib = Dlib()
    print(dlib[0]['keypoints'].shape)