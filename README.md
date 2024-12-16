# Introduction

DISegmentation is an open source  Dichotomous Image Segmentation toolbox based on Jittor.

#### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various dichotomous image segmentation methods. Currently, it includes **4 mainstream models, 10 loss functions, 5 data preprocessing methods, and 10 evaluation metrics.** 

- **Unified Code Structure**

  The code implementations of all methods have been unified into a single framework, making it easier for one to quickly understand the code and intuitively see the differences in the implementation details of various methods.

- **Extremely simple and fast deployment experience**

  Users can deploy the entire process of training, inference, and evaluation with just a few lines of code modification, using a script for one-click execution.

# Get started

#### Environment Setup

```python
conda create -n jittor python=3.7 -y && conda activate jittor
pip install -r requirements.txt
# run Run the following code to successfully install Jittor and enable CUDA.
python3.7 -m jittor.test.test_example
# replace this var with your nvcc location 
export nvcc_path="/usr/local/cuda/bin/nvcc" 
# run a simple cuda test
python3.7 -m jittor.test.test_cuda
# If multi-GPU training is required, you also need to ...
pip install openmpi-bin openmpi-common libopenmpi-dev
# Jittor will automatically check if the environment variable contains mpicc. If mpicc is successfully detected, the following message will be output:
# [i 0502 14:09:55.758481 24 __init__.py:203] Found mpicc(1.10.2) at /usr/bin/mpicc
# If not, the user needs to manually add: export mpicc_path=/your/mpicc/path
```

#### Data Preparation

Download [DIS5K](https://xuebinqin.github.io/dis/index.html) in its official page.

#### Weights Preparation

Download [ISNet](https://github.com/xuebinqin/DIS), [UDUN](https://github.com/PJLallen/UDUN), [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), [MVANet](https://github.com/qianyu-dlut/MVANet/) pretrained model and backbone weights in their official pages.

#### Project Structure

```python
CODE_ROOT/
    ├── DIS
    	├── model  # Implementation of different methods and backbones, currently supporting ISNet, UDUN, BiRefNet, MVANet 
    		├── birefnet
    			├── modules
    			├── birefnet.py
    		├──  backbones
    			├──  swin_v1.py
    	├── evaluation # The code for evaluation metric, currently supporting 10 types, including maxFm, wFmeasure, MAE, Smeasure, meanEm, HCE, etc.
    	├── dataset.py # Create the dataloader required for training
    	├── imgae_proc.py # Data preprocessing methods, currently supporting 5 types: 'color_enhance', 'rotate', 'pepper', 'flip', and 'crop'
    	├── loss.py # Loss functions, currently supporting 10 types including 'bce', 'mae', 'iou', 'ssim', etc., and also supporting multi-loss.
    	├── config.py # Used to modify general settings such as input_size, batch_size, etc., or model-specific settings, with detailed comments explaining each setting.
    	├── eval.py # Run For Evaluation
    	├── inference.py # Run For Inference
    	├── train.py # Run For Train
    	├── train.sh/test.sh/train_test.sh # Used for one-click training, inference, and evaluation.
```

#### Run

First, select the desired model, dataset, file paths, and other basic settings in the config.py. Then, you can run the train_test.sh or modify more settings in config.py. Each setting in config.py is thoroughly explained.

```python
# Train & Test & Evaluation
./train_test.sh METHOD_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
# Example: ./train_test.sh BiRefNet 0,1,2,3,4,5,6,7 0

# See train.sh / test.sh for only training / test-evaluation.
```

#### To do

- More models and more backbones support
- Visualization module
- More great ideas