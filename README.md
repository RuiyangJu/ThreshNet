# ThreshNet: An Efficient DenseNet Using Threshold Mechanism to Reduce Connections

> [ThreshNet: An Efficient DenseNet Using Threshold Mechanism to Reduce Connections](https://arxiv.org/abs/2201.03013)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/threshnet-an-efficient-densenet-using/image-classification-on-svhn)](https://paperswithcode.com/sota/image-classification-on-svhn?p=threshnet-an-efficient-densenet-using)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/threshnet-an-efficient-densenet-using/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=threshnet-an-efficient-densenet-using)

## Abstract
ThreshNet is a network that using a threshold mechanism to further optimize the method of connections. Different numbers of connections for different convolution layers are discarded to speed up inference of the network. ThreshNet has been evaluated with image classification using data sets of CIFAR 10 and SVHN under platforms of NVIDIA RTX 3050 and Raspberry Pi 4. Experimental results show that, compared with HarDNet68, GhostNet, MobileNetV2, ShuffleNet, and EfficientNet, the inference time of the proposed ThreshNet79 is 5%, 9%, 10%, 18%, and 20% faster, respectively. The number of parameters of ThreshNet95 is 55% less than that of HarDNet85.
 
<p align="center">
  <img src="Img/threshnet.jpg" width="640" title="threshnet">
</p>

<p align="center">
 <img src="Img/threshold-mechanism.jpg" width="640" title="threshold-mechanism">
</p>

## Citation
If you find ThreshNet useful in your research, please consider citing:

	@article{ju2022threshnet,
	  title={ThreshNet: An Efficient DenseNet using Threshold Mechanism to Reduce Connections},
	  author={Ju, Rui-Yang and Lin, Ting-Yu and Jian, Jia-Hao and Chiang, Jen-Shiun and Yang, Wei-Bin},
	  journal={IEEE Access},
	  volume={10},
	  pages={82834--82843},
	  year={2022},
	  publisher={IEEE}
	}
	 
## Contents
1. [Introduction](#introduction)
2. [Usage](#Usage)
3. [Results](#Results)
4. [Requirements](#Requirements)
5. [Config](#Config)
6. [References](#References)

## Usage
```bash
python3 main.py
```
optional arguments:

    --lr                default=1e-3    learning rate
    --epoch             default=200     number of epochs tp train for
    --trainBatchSize    default=100     training batch size
    --testBatchSize     default=100     test batch size

## Results
| Name | GPU Time (ms) | C10 Error (%) | FLOPs (G) | MAdd (G) | Memory (MB) | #Params (M) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ThreshNet28** | 0.35 | 14.75 | 2.28 | 4.55 | 83.26 | 10.18 |
| SqueezeNet | 0.36 | 14.25 | 2.69 | 5.32 | 211.42 | 0.78 |
| MobileNet | 0.38 | 16.12 | 2.34 | 4.63 | 230.84 | 3.32 |
| **ThreshNet79** | 0.42 | 13.66  | 3.46 | 6.90 | 109.68  | 14.31 |
| HarDNet68 | 0.44 | 14.66 | 4.26 | 8.51 | 49.28 | 17.57 |
| MobileNetV2 | 0.46 | 14.06 | 2.42 | 4.75 | 384.78 | 2.37 |
| **ThreshNet95** | 0.46 | 13.31 | 4.07 | 8.12 | 132.34 | 16.19 | 
| HarDNet85 | 0.50 | 13.89 | 9.10 | 18.18 | 74.65 | 36.67 |

\* GPU Time is the inference time per image on NVIDIA RTX 3050

## Requirements
* Python 3.6+
* Pytorch 0.4.0+
* Pandas 0.23.4+
* NumPy 1.14.3+

## Config
###### Optimizer 
__Adam Optimizer__
###### Learning Rate
__1e-3__ for [1,74] epochs <br>
__5e-4__ for [75,149] epochs <br>
__2.5e-4__ for [150,200) epochs <br>


## References
* [torchstat](https://github.com/Swall0w/torchstat)
* [pytorch-cifar10](https://github.com/soapisnotfat/pytorch-cifar10)
* [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet)
