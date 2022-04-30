# 6.869 Computer Vision Final Project (Spring 2022)
By Brian Lee and Srinath Mahnakali

## Description
Pytorch implementation for training neural implicit representations for single-image
super-resolution. Compares the following three upsampling models on a downsampled subset 
of the [FFHQ1024 Dataset](https://github.com/NVlabs/ffhq-dataset):

* Bilinear interpolation
* SRCNNs using the implementation gvien [here](https://github.com/yjn870/SRCNN-pytorch)
* Training a Neural Field and then sub-sampling with smaller intervals

## Required Packages
Make sure you have unrar installed. On mac, run 
```
brew install rar
```

## Generating Data
To replicate the experiment, run generator.py as follows with the random seed 123:
```
python3 generator.py --dsize 120 --train_val_test_size 100 10 10 --random_seed 123 output_folder ./data
```
The corresponding images will be stored in the 'data' folder. 

## Training the SRCNN

## Training the Neural Field

## Recreating Experiments

## Miscallaneous Issues:

### To fix g++ issues on satori
#### if you get some c++ compiler warning
```
export CXX=g++
```
#### cuda home not set
```
export CUDA_HOME=/software/cuda/11.4
```
#### libcudart.so.11.0 issues
```
export LD_LIBRARY_PATH=/software/cuda/11.4/targets/ppc64le-linux/lib/
```