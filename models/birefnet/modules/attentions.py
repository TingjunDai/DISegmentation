import jittor as jt
from jittor import init
from jittor import nn
import numpy as np

jt.flags.use_cuda = 1

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv(channels, (channels // reduction), 1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv((channels // reduction), channels, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = nn.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight


class PSA(nn.Module):

    def __init__(self, in_channels, S=4, reduction=4):
        super().__init__()
        self.S = S
        _convs = []
        for i in range(S):
            _convs.append(nn.Conv((in_channels // S), (in_channels // S), ((2 * (i + 1)) + 1), padding=(i + 1)))
        self.convs = nn.ModuleList(_convs)
        self.se_block = SEWeightModule((in_channels // S), reduction=(S * reduction))
        self.softmax = nn.Softmax(dim=1)

    def execute(self, x):
        (b, c, h, w) = x.shape
        SPC_out = x.view((b, self.S, (c // self.S), h, w))
        for (idx, conv) in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :].clone())
        se_out = []
        for idx in range(self.S):
            se_out.append(self.se_block(SPC_out[:, idx, :, :, :]))
        SE_out = jt.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)
        softmax_out = self.softmax(SE_out)
        PSA_out = (SPC_out * softmax_out)
        PSA_out = PSA_out.view((b, (- 1), h, w))
        return PSA_out


class SGE(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = jt.array(jt.zeros((1, groups, 1, 1)))
        self.bias = jt.array(jt.zeros((1, groups, 1, 1)))
        self.sig = nn.Sigmoid()

    def execute(self, x):
        (b, c, h, w) = x.shape
        x = x.view(((b * self.groups), (- 1), h, w))
        xn = (x * self.avg_pool(x))
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(((b * self.groups), (- 1)))
        t = (t - t.mean(dim=1, keepdim=True))
        std = (t.std(dim=1, keepdim=True) + 1e-05)
        t = (t / std)
        t = t.view((b, self.groups, h, w))
        t = ((t * self.weight) + self.bias)
        t = t.view(((b * self.groups), 1, h, w))
        x = (x * self.sig(t))
        x = x.view((b, c, h, w))
        return x