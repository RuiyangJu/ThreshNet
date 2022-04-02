# CNN architecture parameters
Measure the parameters of different CNN model architectures by torchstat.
CNN includes: ResNet, ResNeXt, Wide_ResNet, DenseNet, ThreshNet, HarDNet.

### Result
Name	              Params (M)	MAdds (G)	FLOPs (G)	Memory (MB)	MemR+W (MB)
ResNet-50	         25.55 	       8.22 	 4.12 	 109.69 	 317.43 
ResNet-101	         44.54 	       15.66 	 7.84 	 161.75 	 494.00 
ResNet-152	         60.19 	       23.11 	 11.57 	 226.06 	 682.30 
ResNeXt-50-32x4d	   25.02 	       8.51 	 4.27 	 134.76 	 365.56 
ResNeXt-101-32x8d	   88.79 	       32.93 	 16.49 	 276.02 	 891.31 
Wide_ResNet-50-2	   68.88 	       22.85 	 11.43 	 134.76 	 532.85 
WIde_ResNet-101-2	   126.88 	       45.58 	 22.81 	 199.84 	 884.28 
DenseNet-121	   7.97 	       5.74 	 2.88 	 147.10 	 359.71 
DenseNet-169	   14.15 	       6.81 	 3.42 	 174.14 	 448.55 
ThreshNet-79	   15.32 	       6.90 	 3.46 	 109.68 	 299.96 
ThreshNet-95	   17.14 	       8.12 	 4.07 	 132.32 	 360.30 
HarDNet-68	         17.57      	 8.51 	 4.26 	 49.28 	 181.97 
HarDNet-85	         36.67 	       18.18 	 9.10 	 74.65 	 313.42 

### Module
```python
from torchstat import stat
import torchvision.models as models

model = models.resnet18()
stat(model, (3, 224, 224))
```

## Requirements
* Python 3.6+
* Pytorch 0.4.0+
* Pandas 0.23.4+
* NumPy 1.14.3+

## References
* [torchstat](https://github.com/Swall0w/torchstat)
* [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet)