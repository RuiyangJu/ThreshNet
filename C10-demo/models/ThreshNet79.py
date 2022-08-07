# Thanks to PingoLH
# The code refers to https://github.com/PingoLH/Pytorch-HarDNet

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel, stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU(True))                                          
    def forward(self, x):
        return super().forward(x)

class ThreshLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super(ThreshLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv1', nn.Conv2d(in_channels, 128, kernel_size=1, stride=stride, bias=bias)),
        self.add_module('norm2', nn.BatchNorm2d(128)),
        self.add_module('relu2', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(128, out_channels, kernel_size=kernel, stride=stride, padding=1, bias=bias)),
    def forward(self, x):
        new_features = super(ThreshLayer, self).forward(x)
        return torch.cat([x, new_features], 1)

class ThreshBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          layers_.append(ConvLayer(inch, outch))
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        self.layers = nn.ModuleList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

class ThreshNet(nn.Module):
    def __init__(self):
        super().__init__()
        first_ch  = [32, 64]
        grmul = 1.7
        
        ch_list = [  128, 192, 288, 480, 960]
        gr       = [  32, 32, 32, 40, 160]
        n_layers = [   6, 8, 12, 16,  4]
        downSamp = [   1,  1,  0,  1,  0]
        
        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer 3×3conv
        self.base.append ( ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2,  bias=False) )
        # Second Layer 3×3conv
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=3) )
        # MaxPool
        self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        ch = first_ch[1]
        
        #First ThreshBlock
        self.base.append ( ThreshLayer(ch, 32, kernel=3, stride=1,  bias=False) )
        for i in range (96, 256, 32):
          self.base.append ( ThreshLayer(i, 32, kernel=3, stride=1,  bias=False) )     
        self.base.append ( ConvLayer(256, ch_list[0], kernel=1) )
        ch = ch_list[0]
        if downSamp[0] == 1:
          self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

        #Second ThreshBlock
        self.base.append ( ThreshLayer(ch, 32, kernel=3, stride=1,  bias=False) )
        for i in range (160, 384, 32):
          self.base.append ( ThreshLayer(i, 32, kernel=3, stride=1,  bias=False) )                
        self.base.append ( ConvLayer(384, ch_list[1], kernel=1) )
        ch = ch_list[1]
        if downSamp[1] == 1:
          self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

        #Third ThreshBlock
        self.base.append ( ThreshLayer(ch, 32, kernel=3, stride=1,  bias=False) )
        for i in range (224, 576, 32):
          self.base.append ( ThreshLayer(i, 32, kernel=3, stride=1,  bias=False) )           
        self.base.append ( ConvLayer(576, ch_list[2], kernel=1) )
        ch = ch_list[2]
        if downSamp[2] == 1:
          self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

        #Fourth ThreshBlock
        blk = ThreshBlock(ch, gr[3], grmul, n_layers[3])
        ch = blk.get_out_ch()
        self.base.append ( blk )
        self.base.append ( ConvLayer(ch, ch_list[3], kernel=1) )
        ch = ch_list[3]
        if downSamp[3] == 1:
          self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

        #Fifth ThreshBlock
        blk = ThreshBlock(ch, gr[4], grmul, n_layers[4])
        ch = blk.get_out_ch()
        self.base.append ( blk )
        self.base.append ( ConvLayer(ch, ch_list[4], kernel=1) )
        ch = ch_list[4]
        if downSamp[4] == 1:
          self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        ch = ch_list[blks-1]
        self.base.append (
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                Flatten(),
                nn.Linear(ch, 10) ))

    def forward(self, x):
        for layer in self.base:
          x = layer(x)
        return x
        
def threshnet79():
    return ThreshNet()
