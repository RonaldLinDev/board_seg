import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import os

## yolo ds as input
class edgefinder_ds:
    def __init__(self,
                 images_dir,
                 labels_dir,
                 image_transform = False):
        self.image_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = image_transform
        
        self.images = os.listdir(self.image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.image_dir[-3] + 'txt'
        image = read_image(os.path.join(self.image_dir, image_path))
        with open(os.path.join(self.labels_dir, label_path)) as f:
            label = f.readline().split(" ")[1:]
            print(label)
            
        return image, label
