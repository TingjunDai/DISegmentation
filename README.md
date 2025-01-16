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

Download [DIS5K](https://xuebinqin.github.io/dis/index.html) in its official page. The dataset structure is as follows:

```python
DATASET_ROOT/
	├── DIS5K
    	├── DIS-TR
        	├──im
            ├──gt
        ├── DIS-VD
        ├── DIS-TE1
        ├── DIS-TE2
        ├── DIS-TE3
        ├── DIS-TE4
```

#### Weights Preparation

Download [ISNet](https://github.com/xuebinqin/DIS), [UDUN](https://github.com/PJLallen/UDUN), [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), [MVANet](https://github.com/qianyu-dlut/MVANet/) pretrained model and backbone weights in their official pages.(Jittor is fully compatible with .pth weight files, so the weight files from the above PyTorch code repository can all be used.)

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
        ├── inference_one.ipynb # For inference on a small number of images.
```

#### Run

First, select the desired model, dataset, file paths, and other basic settings in the config.py. Then, you can run the train_test.sh or modify more settings in config.py. Each setting in config.py is thoroughly explained.

```python
# Train & Test & Evaluation
./train_test.sh METHOD_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
# Example: setsid nohup ./train_test.sh BiRefNet 0,1,2,3,4,5,6,7 0 $>nohup.log $ for multi-GPU training
# or setsid nohup ./train_test.sh BiRefNet 0 0 $>nohup.log $ for single-GPU training
# See train.sh / test.sh for only training / inference-evaluation.
```

#### Fastest way to train a model

All the following code can be modified in the `config.py` file.

```python
# For ISNet                      
# First, switch the model to 'ISNet_GTEncoder', then train it to obtain the weight file with fmeasure > 0.99. 
self.model = ['ISNet', 'UDUN', 'BiRefNet', 'ISNet_GTEncoder', 'MVANet'][3]
# Replace the 'gt_encoder' in backbone_weights with the path to the weight file, and then switch the model to 'ISNet' for training.
self.backbone_weights = {
    …
    'gt_encoder' = 'path'
}
self.model = ['ISNet', 'UDUN', 'BiRefNet', 'ISNet_GTEncoder', 'MVANet'][0]

# For UDUN
# Switch the model to 'UDUN', then replace the 'resnet50' in backbone_weights with the path to the weight file, then run 'utils.py' to load the auxiliary dataset required for model training, and then start training.
self.backbone_weights = {
    …
    'resnet50' = 'path'
}
python utils.py  # Don't forget to run utils.py before start training. 

# For BiRefNet
# Switch the model to 'BiRefNet', BiRefNet supports all backbone networks from 'swin_v1_t' to 'swin_v1_l' and others. Simply change the 'birefnet_bb' to the corresponding network. Don't forget to create the corresponding weight paths in 'backbone_weights'.
self.birefnet_bb = [ # Backbones supported by BiRefNet
            'pvt_v2_b2', 'pvt_v2_b5',  # 0-bs10, 1-bs5
            'swin_v1_b', 'swin_v1_l',  # 2-bs9, 3-bs6
            'swin_v1_t', 'swin_v1_s',  # 4, 5
            'pvt_v2_b0', 'pvt_v2_b1',  # 6, 7
        ][4]

# For MVANet
# Switch the model to 'MVANet', MVANet also supports all backbone networks from 'swin_v1_t' to 'swin_v1_l'. Simply change the 'mva_bb' to the corresponding network. Don't forget to create the corresponding weight paths in 'backbone_weights'.
self.mva_bb = [ # Backbones supported by MVANet
            'swin_v1_b', 'swin_v1_l',  # 0, 1
            'swin_v1_t', 'swin_v1_s',  # 2, 3
        ][0]
```

#### Inference a few images using different models

If you only want to briefly understand the qualitative effects of different models, you can use `inference_one.ipynb` to perform inference on a small number of images. To make it easier for everyone to use, I have organized the weight files for four models([Google Drive](https://drive.google.com/drive/folders/16BrDCH8jdoLV98dENSxRaieyd6jNagHX?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1jHjItfPcs7eMM-MktOsYoQ?pwd=9412)). You only need to download them and replace the `ckpt` folder, and the code will run without needing to adjust anything in `config.py`.

#### To do

- More models and more backbones support
- Visualization module
- More great ideas