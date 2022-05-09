#imports
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, utils
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from constants import num_files, num_coords

class DogData(Dataset):
    '''
    Dataset of Dog Images
    '''
    def __init__(self, data_class: int, set_name, transforms = None, with_coords = False):
        '''
        data_class: 64 or 128 
        transforms: transforms to apply
        set name: set name (test, train, val)
        '''
        assert set_name in ['test', 'train', 'val']
        self.data_class = data_class
        self.img_dir = './data/working/'+str(self.data_class)+"/"+set_name
        
        if not transforms:
            transforms = ToTensor()

        self.transforms = transforms
        self.set_name = set_name
        self.with_coords = with_coords

    def __len__(self):
        if self.set_name == 'train':
            if self.with_coords:
                return num_files(self.data_class, 'train')*(128**2)+num_files(self.data_class, 'val')*(self.data_class**2)\
                    +num_files(self.data_class, 'test')*(self.data_class**2)
            return num_files(self.data_class, 'train')
        elif self.set_name == 'val':
            if self.with_coords: 
                return num_files(self.data_class, 'val')*(self.data_class**2) 
            return num_files(self.data_class, 'val')
        elif self.set_name == 'test':
            if self.with_coords:
                return num_files(self.data_class, 'test')*(self.data_class**2) 
            return num_files(self.data_class, 'test')

    def __getitem__(self, idx: int): 

        if self.with_coords:
            if idx < (128**2)*num_files(self.data_class, 'train'):
                id, rem = divmod(idx, self.data_class)

            # id, rem = divmod(idx, self.data_class**2)
            # image_path = self.img_dir + '/' + self.set_name + '_' + str(id) + '_' + str(self.data_class) + '.jpg'
            # img = Image.open(image_path)
            # x_coord, y_coord = divmod(rem, self.data_class)
            # coord = torch.tensor([x_coord, y_coord]).reshape(2,-1)
            # intensity = torch.tensor(img.getpixel((x_coord, y_coord)))/255

            # if self.transforms:
            #     img = self.transforms(img)
            
            # return ((img, coord), intensity)
        else:
            image_path = self.img_dir + '/' + self.set_name + '_' + str(idx) + '_' + str(self.data_class) + '.jpg'
            img = Image.open(image_path)
            if self.transforms:
                img = self.transforms(img)
            return img


#################
## SRCNN stuff ##
#################

class SRCNNDataset(DogData):
    def __init__(self, transforms, set_name: str):
        '''
        Dataset for SRCNN: consists of two datasets of 64x64 and target 128x128 images
        transforms: transforms to apply
        set name: set name (test, train, val)
        '''
        assert set_name in ['test', 'train', 'val']
        self.img_data = DogData(64, set_name, transforms, with_coords=False)
        self.target_data = DogData(128, set_name, None, with_coords=False)
    
    def __len__(self):
        assert len(self.img_data) == len(self.target_data)
        return len(self.img_data)
    
    def __getitem__(self, idx: int):
        """
        Returns tuple of downscaled image, ground truth high resolution image
        """
        return self.img_data[idx], self.target_data[idx]

def get_srcnn_dataloaders(batch_size=5):
    train_dataset = SRCNNDataset(None, 'train')
    val_dataset = SRCNNDataset(None, 'val')
    test_dataset = SRCNNDataset(None, 'test')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # shuffle training set
        pin_memory=True,  # pin_memory allows faster transfer from CPU to GPU
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
    return dataloaders

if __name__ == '__main__':
    sample = DogData(64, 'train', with_coords=True)
    print(len(sample))
    # srcnn_sample = SRCNNDataset(None, 'train')
    # print(len(srcnn_sample))
