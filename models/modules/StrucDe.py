import jittor as jt
from jittor import init
from jittor import nn

jt.flags.use_cuda = 1

class StrucDe(nn.Module):

    def __init__(self):
        super(StrucDe, self).__init__()
        self.conv0 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm(32)
        self.conv1 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(32)
        self.conv3 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm(32)
        self.conv4 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm(32)
        self.conv4_reduce = nn.Conv(32, 16, 1)
        self.bn4_reduce = nn.BatchNorm(16)
        self.conv5 = nn.Conv(16, 16, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm(16)

    def execute(self, in_feat):
        out0 = nn.relu(self.bn0(self.conv0(in_feat[0])))
        out0_up = nn.interpolate(out0, size=in_feat[1].shape[2:], mode='bilinear')
        out1 = nn.relu(self.bn1(self.conv1((out0_up + in_feat[1]))))
        out1_up = nn.interpolate(out1, size=in_feat[2].shape[2:], mode='bilinear')
        out2 = nn.relu(self.bn2(self.conv2((out1_up + in_feat[2]))))
        out2_up = nn.interpolate(out2, size=in_feat[3].shape[2:], mode='bilinear')
        out3 = nn.relu(self.bn3(self.conv3((out2_up + in_feat[3]))))
        out3_up = nn.interpolate(out3, size=in_feat[4].shape[2:], mode='bilinear')
        out4 = nn.relu(self.bn4(self.conv4((out3_up + in_feat[4]))))
        out4_up = nn.interpolate(out4, size=in_feat[5].shape[2:], mode='bilinear')
        out4_up = nn.relu(self.bn4_reduce(self.conv4_reduce(out4_up)))
        out5 = nn.relu(self.bn5(self.conv5((out4_up + in_feat[5]))))
        return out0, out1, out2, out3, out4, out5



