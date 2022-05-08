import torch
from dataloader import DogData
import torchvision.transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time
from tqdm import tqdm
import torchvision.models as models
import copy

batch_size = 1 
num_epochs = 10
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_encoder(num_classes, feature_extract, use_pretrained = True):
    '''
    Initializes a ResNet model with a given number of classes. 
    
    params:
    @num_classes: defines number of classes to use for the model
    @use_pretrained: defines whether or not to use pretrained weights
    in training phase. Defaults to True.
    '''
    #fine-tuned model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft

class NeuralField(nn.Module):
    def __init__(self, encoder):
        super(NeuralField, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(130, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
        self.encoder = encoder
    
    def forward(self, x):
        z = self.encoder(x[0]) 
        coords = x[1].view(batch_size, 2)   
        input = torch.cat((z,coords), 1)
        intensity = self.linear_relu_stack(input)
        return intensity

if __name__ == '__main__':
    encoder = initialize_encoder(128, False, True)

    data_transforms = {
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    }
    dataset64 = DogData(64, data_transforms, 'train', True)
    