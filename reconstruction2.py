import torch
from dataloader import DogData
import torchvision.transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time
from tqdm import tqdm
import copy
from neural_field import NeuralField
import matplotlib.pyplot as plt
import argparse
import PIL.Image as Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argparser():   
    '''
    Argparser Setup. 
    
    Arguments: 
        --image_num: number of image to run the reconstruction on in dataset.
    '''

    #initialize argparser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--set_class', type = str, 
                        default = 'test',
                        help = 'name of set')
    parser.add_argument('--image_class', type = str, 
                        default = '64',
                        help='class of image'
                        )
    parser.add_argument('--image_num', type = str, 
                        default = '4',
                        help='number of image'
                        )
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = argparser()
    model = torch.load('./neural_model/chkpts/chkpt_40.pt', map_location=device)
    model.eval()

    invTrans = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    reconstructed = torch.zeros((3,64,64)).to(device)
    image = Image.open(os.path.join('./data/working', config.image_class, config.set_class, 
                        config.set_class+'_'+config.image_num+'_'+config.image_class+'.jpg'))
    image = data_transforms(image)
    image = torch.unsqueeze(image,0).to(device)

    plt.figure(figsize=[8,4]);

    encoder = model.encoder
    linear_relu_stack = model.linear_relu_stack

    z = encoder(image)

    for i in tqdm(range(64)): 
        for j in range(64):
            coordinates = torch.tensor([i,j]).view(1,2).to(device)
            input = torch.cat((z,coordinates),1)
            output = linear_relu_stack(input).squeeze()
            reconstructed[:,j,i] = output[:] 
    
    high_quality_image = Image.open(os.path.join('./data/working', '128', config.set_class, 
                        config.set_class+'_'+config.image_num+'_128.jpg')) 
    high_quality_image = transforms.ToTensor()(high_quality_image)
    
    image = invTrans(image)
    plt.subplot(121); plt.imshow(image.squeeze().permute(1,2,0).detach().numpy()); plt.title('Original Image')
    plt.subplot(122); plt.imshow(reconstructed.squeeze().permute(1,2,0).detach().numpy()); plt.title('Reconstructed Image')
    plt.show()