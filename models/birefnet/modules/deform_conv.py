import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.birefnet.modules.dcn_v2 import dcn_v2_conv

class DeformableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        assert ((type(kernel_size) == tuple) or (type(kernel_size) == int))
        kernel_size = (kernel_size if (type(kernel_size) == tuple) else (kernel_size, kernel_size))
        self.stride = (stride if (type(stride) == tuple) else (stride, stride))
        self.padding = (padding if (type(padding) == tuple) else (padding, padding))
        self.offset_conv = nn.Conv(in_channels, ((2 * kernel_size[0]) * kernel_size[1]), kernel_size, stride=stride, padding=self.padding, bias=True)
        init.constant_(self.offset_conv.weight, value=0.0)
        init.constant_(self.offset_conv.bias, value=0.0)
        self.modulator_conv = nn.Conv(in_channels, ((1 * kernel_size[0]) * kernel_size[1]), kernel_size, stride=stride, padding=self.padding, bias=True)
        init.constant_(self.modulator_conv.weight, value=0.0)
        init.constant_(self.modulator_conv.bias, value=0.0)
        self.regular_conv = nn.Conv(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias)
        if bias == False:
            self.bias = np.zeros((out_channels,))
        else:
            self.bias = self.regular_conv.bias
    def execute(self, x):
        offset = self.offset_conv(x)
        modulator = (2.0 * self.modulator_conv(x).sigmoid())
        x = dcn_v2_conv(x, offset, modulator, self.regular_conv.weight, jt.array(self.bias), self.stride, self.padding, (1, 1), 1)
        # x = self.regular_conv(x)
        return x