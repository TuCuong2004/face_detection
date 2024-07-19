import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import tarfile
import zipfile
class Ibug(Dataset):
    def __init__(self, root_dir=r'data/ibug'):
        
        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            print("data don't exist")
            os.makedirs(self.root_dir)
            unzip_data(self.root_dir)
        

def unzip_data(root_dir):
    with zipfile.ZipFile(r'data/ibug.zip', "r") as zip:
        zip.extractall(root_dir)

    
        
            
if __name__ == "__main__":
    cg = Cg()
    print("done")