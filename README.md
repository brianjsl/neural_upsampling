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
python3 generator.py --raw_path ./data/raw --data_path ./data/working --train_val_test_size [50, 10, 10]
```
The corresponding images will be stored in the 'data' folder. You can alter the data size and other
parameters by adding additional flags. The images should be stored in ./data/working. 

## Training the SRCNN and FSRCNN
To train the SRCNN, create a directory 'srcnn_model'. In srcnn_train.py, set `use_srcnn` to `True` and `use_fsrcnn` to `False` 
Then run srcnn_train.py:
```
python3 srcnn_train.py 
```
The directory will contain two model weights which can be used later, 'weights_best_val_acc.pt' and 'weights_last.pt'.

To train the FSRCNN, create a directory 'fsrcnn_model'. In srcnn_train.py, set `use_srcnn` to `False` and `use_fsrcnn` to `True`.
Then run srcnn_train.py. The directory will contain two model weights which can be used later, 'weights_best_val_acc.pt' and 'weights_last.pt'.

## Training the Neural Field
To train the Neural Field, change the `batch_size` paramter in `neural_field.py` to 32 and then run
```
python3 neural_field.py
```
The checkpoints will be stored in the folder `./neural_model/chkpts`. Checkpoint 40 for our own training can be found
[here](https://www.dropbox.com/s/6akc6tt51ht330y/chkpt_40.pt?dl=0). To train on Satori, run 
```
sbatch neural_field_train.slurm
```
in the command line. 

## Recreating Experiments
Once the SRCNN is trained, run srcnn_experiments.py:
```
python3 srcnn_experiments.py
```
It will print the average time, MSE, and PSNR of the SRCNN and bilinear interpolation.

Once the Neural Field is trained, run 
```
python3 neural_field_experiments.py
```
To do reconstruction and visualize images run `reconstruction.py` as follows:
```
python3 reconstruction.py set_class train image_class 128 image_num 4
```
where you replace "train" with whatever class you want to reconstruct (train/val/test) and image_class represents 
whether you want to reconstruct the low-quality images (64) or high-quality images (128) and 4 can be replaced with 
whatever image id you wish to reproduce.
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
