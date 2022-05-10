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
                        default = 'train',
                        help = 'name of set')
    parser.add_argument('--image_class', type = int, 
                        default = 64,
                        help='class of image'
                        )
    parser.add_argument('--image_num', type = str, 
                        default = '17',
                        help='number of image'
                        )
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = argparser()
    model = torch.load('./neural_model/chkpts/chkpt_21.pt', map_location=device)
    model.eval()

    data_transforms = transforms.Compose([
                transforms.ToTensor(),
    ])
    reconstructed = torch.zeros((3,config.image_class,config.image_class)).to(device)
    image = Image.open(os.path.join('./data/working', '64', config.set_class, 
                        config.set_class+'_'+config.image_num+'_64.jpg'))
    image = data_transforms(image)
    image = torch.unsqueeze(image,0).to(device)

    plt.figure(figsize=[8,4]);

    encoder = model.encoder
    linear_relu_stack = model.linear_relu_stack

    z = encoder(image)

    step = 0.5 if config.image_class == 128 else 1
    for i in tqdm(range(config.image_class)): 
        for j in range(config.image_class): 
            coordinates = torch.tensor([step*i,step*j]).view(1,2).to(device)
            input = torch.cat((z,coordinates),1)
            output = linear_relu_stack(input).squeeze()
            reconstructed[:,j,i] = output[:] 
    
    high_quality_image = Image.open(os.path.join('./data/working', '128', config.set_class, 
                        config.set_class+'_'+config.image_num+'_128.jpg')) 
    high_quality_image = transforms.ToTensor()(high_quality_image)
    
    plot_image = high_quality_image if config.image_class == 128 else image
    # image = invTrans(image)
    plt.subplot(121); plt.imshow(plot_image.squeeze().permute(1,2,0).detach().numpy()); plt.title('Original Image')
    plt.subplot(122); plt.imshow(reconstructed.squeeze().permute(1,2,0).detach().numpy()); plt.title('Reconstructed Image')
    plt.show()