import os
import argparse
from tqdm import tqdm
import numpy as np
import subprocess
import shutil


data_keys = ['Linnaeus_dog_64X64', 'Linnaeus_dog_256X256']
data_urls = {'Linnaeus_dog_64X64': 'https://www.dropbox.com/s/9juhj4pdepaev27/Linnaeus_dog_64.zip?dl=0',
             'Linnaeus_dog_256X256': 'https://www.dropbox.com/s/44ui4lt2ah8ve4g/Linnaeus_dog_256.zip?dl=0'}
data_names = {'Linnaeus_dog_64X64': '64', 'Linnaeus_dog_256X256': '256'}

def check_dataset_dir(data_dir):
    '''
    Check if the dataset is downloaded.
    '''
    downloaded = np.all([os.path.isdir(os.path.join(data_dir, key)) for key in data_keys])
    if not downloaded:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_dataset(data_dir)
    else:
        print("Dataset was Found")

def download_dataset(data_dir):
    '''
    Downloads the dataset.
    '''
    for k in data_keys:
        name = k + '.zip'
        url = data_urls[k]
        target_path = os.path.join(data_dir, name)
        cmd = ['curl', '-L', '-o', target_path, url]
        print('Downloading ', name)
        subprocess.call(cmd)
        os.makedirs(os.path.join(data_dir, k))
        cmd = ['unzip', target_path, '-d', os.path.join(data_dir, k)]
        print('Unzip ', name)
        subprocess.call(cmd)


def generator(config):
    '''
    Generates the images given 
    '''
    #Download Raw and/or check if it exists
    check_dataset_dir(config.raw_path)

    #Create Required Directories if not Already Existing
    if not (os.path.isdir(config.data_path)):
        os.makedirs(config.data_path)
        for key in data_keys:
            os.makedirs(os.path.join(config.data_path, data_names[key]))
            for i in ['train', 'val', 'test']:
                os.makedirs(os.path.join(config.data_path, data_names[key], i))
    #split the dataset. 
    rs = np.random.RandomState(config.random_seed) #generate random state
    num_images = 1200
    image_ids = list(np.array(range(1,num_images+1)))
    rs.shuffle(image_ids)

    #dataset sizes
    num_train, num_val, num_test = config.train_val_test_size

    train_ids = image_ids[:num_train]
    val_ids = image_ids[num_train:num_train+num_val]
    test_ids = image_ids[num_train+num_val: num_train+num_val+num_test]

    for k in data_keys:
        for ids in [train_ids, val_ids, test_ids]:
            for new_id, id in enumerate(ids):
                im_name = str(id)+'_'+data_names[k]+'.jpg'
                new_im_name = str(new_id)+'_'+data_names[k]+'.jpg'
                image = os.path.join(config.raw_path, k,'dog', im_name)
                if ids == train_ids:
                    shutil.copy(image, os.path.join(config.data_path, data_names[k], 'train'))
                    os.rename(os.path.join(config.data_path, data_names[k], 'train',im_name), \
                        os.path.join(config.data_path, data_names[k], 'train',new_im_name))
                elif ids == val_ids:
                    shutil.copy(image, os.path.join(config.data_path, data_names[k], 'val'))
                    os.rename(os.path.join(config.data_path, data_names[k], 'val',im_name), \
                        os.path.join(config.data_path, data_names[k], 'val',new_im_name))
                elif ids == test_ids:
                    shutil.copy(image, os.path.join(config.data_path, data_names[k], 'test')) 
                    os.rename(os.path.join(config.data_path, data_names[k], 'test',im_name), \
                        os.path.join(config.data_path, data_names[k], 'test',new_im_name))

def argparser():   
    '''
    Argparser Setup. 
    
    Arguments: 
        --raw_path: path to dataset
        --data_path: path to working dataset
        --train_val_test_size size of train to val to test
        --random_seed: default random seed to choose    
    '''

    #initialize argparser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--raw_path', type = str, 
                        default = './data/raw',
                        help='path to *.zip files'
                        )
    parser.add_argument('--data_path', type = str,
                        default = './data/working',
                        help = 'path for working data direcotry'
                        )
    parser.add_argument('--train_val_test_size', type=int, nargs='+',
                        default=[100, 10, 10], help='Train/Val/Test size')
    parser.add_argument('--random_seed', type=int, default=123)
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = argparser()
    assert len(config.train_val_test_size) == 3
    generator(config)
    