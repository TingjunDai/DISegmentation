from jittor import nn
import jittor as jt

jt.flags.use_cuda = 1

class BasicLatBlk(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicLatBlk, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1, stride=1, padding=0)

    def execute(self, x):
        x = self.conv(x)
        return x
