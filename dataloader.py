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
from constants import num_files, num_coords


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
            return num_files(self.data_class, 'train')
        elif self.set_name == 'val':
            return num_files(self.data_class, 'val')
        elif self.set_name == 'test':
            return num_files(self.data_class, 'test')

    def __getitem__(self, idx: int):
        if self.with_coords:
            pass
        else:
            pass

if __name__ == '__main__':
    sample = DogData(64, None, 'train', False)
    print(len(sample))