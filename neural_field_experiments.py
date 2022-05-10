import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from dataloader import SRCNNDataset, get_srcnn_dataloaders
from srcnn_model import SRCNN
from fsrcnn_model import FSRCNN
from torchvision import datasets, models, transforms
from time import time
from neural_field import NeuralField

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('./neural_model/chkpts/chkpt_40.pt', map_location=device)
model.eval()

def reconstruct(low_res):
    '''
    Reconstructs the high resolution version of an image 
    using the Neural Field from a low resolution image.
    '''

    reconstructed = torch.zeros((3,128, 128)).to(device)
    image = low_res

    encoder = model.encoder
    linear_relu_stack = model.linear_relu_stack

    z = encoder(image)

    step = 0.5
    for i in range(128): 
        for j in range(128): 
            coordinates = torch.tensor([step*i,step*j]).view(1,2).to(device)
            input = torch.cat((z,coordinates),1)
            output = linear_relu_stack(input).squeeze()
            reconstructed[:,j,i] = output[:]  

    return reconstructed.detach()

mse_loss = nn.MSELoss()

def get_neural_field_statistics(low_res, high_res):
    start_time = time()
    predicted = reconstruct(low_res)
    end_time = time()
    amount_time = end_time-start_time
    MSE = mse_loss(predicted, high_res)
    PSNR = 10 * torch.log10(1. / MSE)
    return np.array([amount_time, MSE.item(), PSNR.item()])

if __name__ == '__main__':
    test_dataset = SRCNNDataset(None, 'train')
    num_images = len(test_dataset)
    neural_field_stats_list = np.empty([num_images,3])
    for image_id in range(num_images):
        low_res = test_dataset[image_id][0]
        test = test_dataset[image_id][1]
        neural_field_stats = get_neural_field_statistics(low_res, test)
        neural_field_stats_list[image_id] = neural_field_stats
    avg_neural_field_stats = np.mean(neural_field_stats_list, 0)
    print(avg_neural_field_stats)