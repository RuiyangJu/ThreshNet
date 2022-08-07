# [ThreshNet: An Efficient DenseNet Using Threshold Mechanism to Reduce Connections](https://arxiv.org/abs/2201.03013)
<p align="center">
  <img src="Img/threshnet.jpg" width="640" title="threshnet">
</p>

## Citation
If you find ThreshNet useful in your research, please consider citing:

	@article{ThreshNet 2022,
	 title={ThreshNet: An Efficient DenseNet using Threshold Mechanism to Reduce Connections},
	 author={Rui-Yang Ju, Ting-Yu Lin, Jia-Hao Jian, Jen-Shiun Chiang, Wei-Bin Yang},
	 journal={IEEE Access},
	 year={2022}
	 }
	 
## Contents
1. [Introduction](#introduction)
2. [Usage](#Usage)
2. [Results](#Results)
3. [Requirements](#Requirements)
4. [Config](#Config)
5. [References](#References)

## Introduction
ThreshNet is a network that using a threshold mechanism to further optimize the method of connections. Different numbers of connections for different convolution layers are discarded to speed up inference of the network. ThreshNet has been evaluated with image classification using data sets of CIFAR 10 and SVHN under platforms of NVIDIA RTX 3050 and Raspberry Pi 4. Experimental results show that, compared with HarDNet68, GhostNet, MobileNetV2, ShuffleNet, and EfficientNet, the inference time of the proposed ThreshNet79 is 5%, 9%, 10%, 18%, and 20% faster, respectively. The number of parameters of ThreshNet95 is 55% less than that of HarDNet85.

 <img src="Img/threshold-mechanism.jpg" width="640" title="threshold-mechanism">

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
| Name | GPU Time (ms) | Error (%) | FLOPs (G) | MAdd (G) | Memory (MB) | #Params (M) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-50  | 25.55  | 8.22 | 4.12 | 109.69 | 317.43 |
| ResNet-101  | 44.54  | 15.66 | 7.84 | 161.75 | 494.00 |
| ResNet-152  | 60.19  | 23.11 | 11.57 | 226.06 | 682.30 |
| ResNeXt-50-32x4d  | 25.02  | 8.51 | 4.27 | 134.76 | 365.56 |
| ResNeXt-101-32x8d  | 88.79  | 32.93 | 16.49 | 276.02 | 891.31 |
| Wide_ResNet-50-2  | 68.88  | 22.85 | 11.43 | 134.76 | 532.85 |
| WIde_ResNet-101-2  | 126.88  | 45.58 | 22.81 | 199.84 | 884.28 |
| DenseNet-121  | 7.97  | 5.74 | 2.88 | 147.10 | 359.71 |
| DenseNet-169  | 14.15  | 6.81 | 3.42| 174.14  | 448.55 |
| **ThreshNet-79** | 15.32  | 6.90 | 3.46 | 109.68  | 299.96 |
| **ThreshNet-95** | 17.14  | 8.12  | 4.07 | 132.32  | 360.30 | 
| HarDNet-68  | 17.57  | 8.51 | 4.26 | 49.28 | 181.97 |
| HarDNet-85  | 36.67  | 18.18 | 9.10 | 74.65  | 313.42 |

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
