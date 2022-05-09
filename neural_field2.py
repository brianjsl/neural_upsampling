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
num_epochs = 40
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
            nn.Linear(512,3),
            nn.Sigmoid()
        )
        self.encoder = encoder
    
    def forward(self, x):
        z = self.encoder(x[0]) 
        coords = x[1].view(batch_size, 2)   
        input = torch.cat((z,coords), 1)
        intensity = self.linear_relu_stack(input)
        return intensity

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    '''
    Trains the model for given number of epochs.

    params:
    @ model: model to train
    @ dataloaders: dictionary of dataloaders
    @ criterion: loss function/criterion
    @ optimizer: optimizer used 
    @ num_epochs: number of epochs to train for
    '''
    since = time.time() 
    
    #weights of best model
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('_'*10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            #keep track of losses and corrects        
            running_loss = 0.0

            #Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                labels = labels.to(device).view(batch_size,-1)
                
                #zero the parameter gradients
                optimizer.zero_grad()

                #forward and track history if train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model((inputs[0].to(device), inputs[1].to(device)))
                    loss = criterion(outputs, labels.float())
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
                #statistics to keep track of
                running_loss += loss.item()
                
            epoch_loss = running_loss 

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        torch.save(model, './neural_model/chkpts/chkpt_{}.pt'.format(epoch+1))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    encoder = initialize_encoder(128, False, True)
    model = NeuralField(encoder).to(device)

    print('Initializing Datasets and Dataloaders...')

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_datasets = {x: DogData(64, x, data_transforms, True) for x in ['train', 'val']}

    dataloaders_dict= {x: DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True, 
                            num_workers=2) for x in ['train', 'val']}

    print("Done Initializing Data.")

    #optimizers
    optimizer_nf =  optim.Adam(model.parameters(), lr = learning_rate)

    criterion = nn.MSELoss()

    model = train_model(model, dataloaders_dict, criterion, optimizer_nf, 
                        num_epochs=num_epochs)
    torch.save(model, 'neural_field.pt')


    
   
    