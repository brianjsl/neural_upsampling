# 6.869 Computer Vision Final Project (Spring 2022)
By Brian Lee and Srinath Mahnakali

## Description
Pytorch implementation for training neural implicit representations for single-image
super-resolution. Compares the following three upsampling models on a downsampled subset 
of the 'dog' class in the [Linnaeus 5 dataset](http://chaladze.com/l5/):

* Bilinear interpolation
* SRCNNs using the implementation gvien [here](https://github.com/yjn870/SRCNN-pytorch)
* Training a Neural Field and then sub-sampling with smaller intervals

## Generating Data
To replicate the experiment, run generator.py as follows with the random seed 123:
```
python3 generator.py 
```
The corresponding images will be stored in the 'data' folder. You can alter the data size and other
parameters by adding additional flags. The images should be stored in ./data/working. 

## Training the SRCNN
Create a directory 'srcnn_model', then run srcnn_train.py:
```
python3 srcnn_train.py 
```
The directory will contain two model weights which can be used later, 'weights_best_val_acc.pt' and 'weights_last.pt'.

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
