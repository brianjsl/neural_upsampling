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

#device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srcnn_model = SRCNN()
srcnn_model.load_state_dict(torch.load('srcnn_model/weights_best_val_acc.pt', map_location='cpu'))
fsrcnn_model = FSRCNN(2)
fsrcnn_model.load_state_dict(torch.load('fsrcnn_model/weights_best_val_acc.pt', map_location='cpu'))

#parameters
run_experiments = False
visualize_upsample = True
visualize_id = 9

#get dataloaders
all_dataloaders = get_srcnn_dataloaders()

#MSE Loss
mse_loss = nn.MSELoss()

def upsample_image(model, image, use_fsrcnn=False):
    """
    Upsamples a 64x64 image to a 128x128 image.
    model: the srcnn model to be used
    image: the image to upsample. it is a torch tensor
    
    Returns: an upsampled torch tensor.
    """
    model.eval()
    batch_image = image[None]
    if not use_fsrcnn:
        preprocessed_image = F.interpolate(batch_image, scale_factor=2, mode='bicubic', align_corners=False)
        output = model(preprocessed_image).squeeze()
    else:
        output = model(batch_image).squeeze()
    return output.detach()

def get_srcnn_statistics(model, low_res, high_res, use_fsrcnn=False):
    """
    Computes the performance of a model using our metrics.
    model: the srcnn model to be used
    low_res: the image to upsample. it is a torch tensor
    high_res: the ground truth image. it is a torch tensor

    Returns: amount_time, MSE, and PSNR
    amount_time: the amount of time to upsample the image
    MSE: the mean-squared error of the prediction
    PSNR: the peak-signal-to-noise ratio
    """
    start_time = time()
    predicted = upsample_image(model, low_res, use_fsrcnn)
    end_time = time()
    amount_time = end_time - start_time
    MSE = mse_loss(predicted, high_res)
    PSNR = 10 * torch.log10(1. / MSE)
    return np.array([amount_time, MSE.item(), PSNR.item()])

def get_bilinear_statistics(low_res, high_res):
    """
    Computes the performance of bilinear interpolation using our metrics.
    low_res: the image to upsample. it is a torch tensor
    high_res: the ground truth image. it is a torch tensor

    Returns: amount_time, MSE, and PSNR
    amount_time: the amount of time to upsample the image
    MSE: the mean-squared error of the prediction
    PSNR: the peak-signal-to-noise ratio
    """
    start_time = time()
    batch_image = low_res[None]
    preprocessed_image = F.interpolate(batch_image, scale_factor=2, mode='bilinear', align_corners=False)
    predicted = preprocessed_image.squeeze()
    end_time = time()
    amount_time = end_time - start_time
    MSE = mse_loss(predicted, high_res)
    PSNR = 10 * torch.log10(1. / MSE)
    return np.array([amount_time, MSE.item(), PSNR.item()])


if __name__ == '__main__':
    srcnn_sample = SRCNNDataset(None, 'val')
    if run_experiments:
        num_images = len(srcnn_sample)
        fsrcnn_stats_list, srcnn_stats_list, bilin_stats_list = np.empty([num_images,3]), np.empty([num_images,3]), np.empty([num_images,3])
        for image_id in range(num_images):
            low_res = srcnn_sample[image_id][0]
            test = srcnn_sample[image_id][1]
            srcnn_stats = get_srcnn_statistics(srcnn_model, low_res, test)
            fsrcnn_stats = get_srcnn_statistics(fsrcnn_model, low_res, test, use_fsrcnn=True)
            bilin_stats = get_bilinear_statistics(low_res, test)
            srcnn_stats_list[image_id] = srcnn_stats
            fsrcnn_stats_list[image_id] = fsrcnn_stats
            bilin_stats_list[image_id] = bilin_stats
        avg_srcnn_stats = np.mean(srcnn_stats_list, 0)
        avg_fsrcnn_stats = np.mean(fsrcnn_stats_list, 0)
        avg_bilin_stats = np.mean(bilin_stats_list, 0)
        print(avg_srcnn_stats)
        print(avg_fsrcnn_stats)
        print(avg_bilin_stats)
    elif visualize_upsample:
        image_id = visualize_id
        #low-res image
        low_res = srcnn_sample[image_id][0]
        plt.imshow(low_res.permute(1,2,0))
        plt.show()
        #low-res image interpolated
        batch_image = low_res[None]
        preprocessed_image = F.interpolate(batch_image, scale_factor=4, mode="bilinear", align_corners=False)
        bilinear_image = preprocessed_image.squeeze()
        plt.imshow(bilinear_image.permute(1,2,0))
        plt.show()
        #upsampled srcnn image
        upsampled_srcnn = upsample_image(srcnn_model, low_res)
        plt.imshow(upsampled_srcnn.permute(1,2,0))
        plt.show()
        #upsampled fsrcnn image
        upsampled_fsrcnn = upsample_image(fsrcnn_model, low_res, use_fsrcnn=True)
        plt.imshow(upsampled_fsrcnn.permute(1,2,0))
        plt.show()
        #ground truth image
        test = srcnn_sample[image_id][1]
        plt.imshow(test.permute(1,2,0))
        plt.show()