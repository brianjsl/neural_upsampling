#imports
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class DogData(Dataset):
    '''
    Dataset of Dog Images
    '''
    def __init__(self, data_class: int, transforms, set_name: str, with_coords = False):
        '''
        data_class: 64 or 256 
        transforms: transforms to apply
        set name: set name (test, train, val)
        '''
        assert set_name in ['test', 'train', 'val']
        self.data_class = data_class
        self.img_dir = './data/working/'+str(self.data_class)+"/"+set_name
        self.transforms = transforms
        self.set_name = set_name
        self.with_coords = with_coords

    def __len__(self):
        if self.set_name == 'train':
            if self.with_coords:
                
        elif self.set_name == 'val':
            pass
        elif self.set_name == 'test':
            pass

    def __getitem__(self):
        pass
