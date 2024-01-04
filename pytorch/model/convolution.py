import numpy as np
import torch
from torch import nn
from torch.nn import init
from .registry import Registry
from collections import OrderedDict
import math
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple

class NonSquareKernel(nn.Module):

    def __init__(self, channel=3, output=45, height=9, width=1):
        super().__init__()
        self.conv0 = nn.Conv2d(channel, output, kernel_size=9*5)
        self.conv1 = nn.Conv2d(output, output, kernel_size=(height, width))
        self.conv2 = nn.Conv2d(output, output, kernel_size=(height, width), padding=(4, 0), stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(3, 4), stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.batch_norm = nn.BatchNorm2d(output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        concat1 = x
        for i in range (4):
            x = self.conv2(x)
            x = self.relu(x)
            x = self.batch_norm(x)
            if(i%2 != 0):
                x += concat1
                concat1 = x
        
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.softmax(x)

        return x

class FFTResBlock(nn.Module):

    def __init__(self, out_channel, norm='backward'):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding = 3 // 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding = 3 // 2)
        )
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1)
        )
        self.dim = out_channel
        self.norm = norm
    
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        return self.main(x) + x + y