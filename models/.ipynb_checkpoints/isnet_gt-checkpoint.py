import jittor as jt
from jittor import init
from jittor import nn

from models.modules.RSU import RSU4, RSU5, RSU6, RSU7, RSU4F


def _upsample_like(src, tar):
    src = nn.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    return src


class myrebnconv(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(myrebnconv, self).__init__()
        self.conv = nn.Conv(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation,
                            groups=groups)
        self.bn = nn.BatchNorm(out_ch)
        self.rl = nn.ReLU()

    def execute(self, x):
        return self.rl(self.bn(self.conv(x)))


class ISNetGTEncoder(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(ISNetGTEncoder, self).__init__()
        self.conv_in = myrebnconv(in_ch, 16, 3, stride=2, padding=1)
        self.stage1 = RSU7(16, 16, 64)
        self.pool12 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage4 = RSU4(128, 32, 256)
        self.pool45 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage5 = RSU4F(256, 64, 512)
        self.pool56 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage6 = RSU4F(512, 64, 512)
        self.side1 = nn.Conv(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv(512, out_ch, 3, padding=1)

    def execute(self, x):
        hx = x
        hxin = self.conv_in(hx)
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        d1 = self.side1(hx1)
        d1 = _upsample_like(d1, x)
        d2 = self.side2(hx2)
        d2 = _upsample_like(d2, x)
        d3 = self.side3(hx3)
        d3 = _upsample_like(d3, x)
        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, x)
        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, x)
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)
        return [d1, d2, d3, d4, d5, d6], [hx1, hx2, hx3, hx4, hx5, hx6]  # d为未经过sigmoid的最终层，hx为上一层
