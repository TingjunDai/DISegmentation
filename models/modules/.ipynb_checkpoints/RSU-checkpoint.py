import jittor as jt
from jittor import init
from jittor import nn

jt.flags.use_cuda = 1

class REBNCONV(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv(in_ch, out_ch, 3, padding=(1 * dirate), dilation=(1 * dirate), stride=stride)
        self.bn_s1 = nn.BatchNorm(out_ch)
        self.relu_s1 = nn.ReLU()

    def execute(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout


def _upsample_like(src, tar):
    src = nn.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    return src


class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        (b, c, h, w) = x.shape
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(jt.contrib.concat((hx7, hx6), dim=1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(jt.contrib.concat((hx6dup, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(jt.contrib.concat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(jt.contrib.concat((hx6, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(jt.contrib.concat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(jt.contrib.concat((hx5, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4, hx3), dim=1))
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3d, hx2), dim=1))
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2d, hx1), dim=1))
        return hx1d + hxin