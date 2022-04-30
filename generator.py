import os
import argparse
from tqdm import tqdm
import numpy as np
import subprocess

data_keys = ['Linnaeus_dog_64X64', 'Linnaeus_dog_256X256']
data_urls = {'Linnaeus_dog_64X64': 'https://www.dropbox.com/s/9juhj4pdepaev27/Linnaeus_dog_64.zip?dl=0',
             'Linnaeus_dog_256X256': 'https://www.dropbox.com/s/44ui4lt2ah8ve4g/Linnaeus_dog_256.zip?dl=0'}

def check_dataset_dir(data_dir):
    '''
    Check if the dataset is downloaded.
    '''
    downloaded = np.all([os.path.isfile(os.path.join(data_dir, key)) for key in data_keys])
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
        name = k + '.rar'
        url = data_urls[k]
        target_path = os.path.join(data_dir, name)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', name)
        subprocess.call(cmd)
        cmd = ['unzip', '-d', target_path]
        print('Unzip ', name)
        subprocess.call(cmd)


def generator(config):
    '''
    Generates the images given 
    '''
    pass

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
    