import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import dataloader
from torchvision import datasets, models, transforms
import time
import copy
import os
from srcnn_model import SRCNN
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get dataloaders
all_dataloaders = dataloader.get_srcnn_dataloaders(batch_size=10)


#train the srcnn
def train_model(model, dataloaders, criterion, optimizer, save_dir = None, save_all_epochs=False, num_epochs=25):
    '''
    model: The NN to train
    dataloaders: A dictionary containing at least the keys 
                 'train','val' that maps to Pytorch data loaders for the dataset
    criterion: The Loss function
    optimizer: The algorithm to update weights 
               (Variations on gradient descent)
    num_epochs: How many epochs to train for
    save_dir: Where to save the best model weights that are found, 
              as they are found. Will save to save_dir/weights_best.pt
              Using None will not write anything to disk
    save_all_epochs: Whether to save weights for ALL epochs, not just the best
                     validation error epoch. Will save to save_dir/weights_e{#}.pt
    '''
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            # TQDM has nice progress bars
            for images, targets in tqdm(dataloaders[phase]):
                images = images.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    preprocessed_images = F.interpolate(images, scale_factor=4, mode='bicubic', align_corners=False)
                    model = model.to(device)
                    outputs = model(preprocessed_images)
                    loss = criterion(outputs, targets)

                    # backprop + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_acc_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_loss)
            if save_all_epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, f'weights_{epoch}.pt'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # save and load best model weights
    torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'weights_last.pt'.format(epoch)))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

def make_optimizer(model, learning_rate, print_parameters=False):
    # Get all the parameters
    params_to_update = model.parameters()
    if print_parameters:
      print("Params to learn:")
      for name, param in model.named_parameters():
          if param.requires_grad == True:
              print("\t",name)

    optimizer = optim.Adam(params_to_update, lr=learning_rate)
    return optimizer

def get_loss():
    # Create an instance of the loss function
    criterion = nn.MSELoss()
    return criterion


if __name__ == '__main__':
    #variables and parameters
    model = SRCNN()
    criterion = get_loss()
    lr = 1e-4
    optimizer = make_optimizer(model, lr)

    model, val_acc_hist, train_acc_hist = train_model(
                                            model=model,
                                            dataloaders=all_dataloaders,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            save_dir='srcnn_model',
                                            num_epochs=30
                                        )