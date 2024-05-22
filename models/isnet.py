import jittor as jt
from jittor import init
from jittor import nn

from models.modules.RSU import RSU4, RSU5, RSU6, RSU7, RSU4F

jt.flags.use_cuda = 1

def _upsample_like(src, tar):
    src = nn.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    return src


class ISNetDIS(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(ISNetDIS, self).__init__()
        self.conv_in = nn.Conv(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
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
        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(jt.contrib.concat((hx6up, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(jt.contrib.concat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)
        return [d1, d2, d3, d4, d5, d6], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
