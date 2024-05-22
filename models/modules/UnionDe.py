import jittor as jt
from jittor import init
from jittor import nn

jt.flags.use_cuda = 1

class TSA(nn.Module):

    def __init__(self):
        super(TSA, self).__init__()
        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv(64, 1, 1)
        self.conv2 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(32)
        self.conv3 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm(32)
        self.conv4 = nn.Conv(64, 32, 1)
        self.bn4 = nn.BatchNorm(32)
        self.conv5 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm(32)

    def execute(self, feat_trunk, feat_struct):
        y = nn.relu(self.bn4(self.conv4(feat_trunk)))
        x = self.act(self.conv1(feat_trunk))
        x = (x * feat_struct)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3((x + feat_struct))))
        y = nn.relu(self.bn5(self.conv5((x + y))))
        return y


class MSA_256(nn.Module):

    def __init__(self):
        super(MSA_256, self).__init__()
        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv(32, 1, 1)
        self.conv2 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(32)
        self.conv3 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm(32)
        self.conv4 = nn.Conv(32, 32, 1)
        self.bn4 = nn.BatchNorm(32)
        self.conv5 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm(32)

    def execute(self, feat_trunk, feat_struct):
        y = nn.relu(self.bn4(self.conv4(feat_trunk)))
        x = self.act(self.conv1(feat_trunk))
        x = (x * feat_struct)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3((x + feat_struct))))
        y = nn.relu(self.bn5(self.conv5((x + y))))
        return y


class MSA_512(nn.Module):

    def __init__(self):
        super(MSA_512, self).__init__()
        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv(32, 1, 1)
        self.conv2 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(32)
        self.conv3 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm(32)
        self.conv4 = nn.Conv(32, 32, 1)
        self.bn4 = nn.BatchNorm(32)
        self.conv5 = nn.Conv(32, 32, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm(32)

    def execute(self, feat_mask, feat_struct):
        y = nn.relu(self.bn4(self.conv4(feat_mask)))
        x = self.act(self.conv1(feat_mask))
        x = (x * feat_struct)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3((x + feat_struct))))
        y = nn.relu(self.bn5(self.conv5((x + y))))
        return y


class MSA_1024(nn.Module):

    def __init__(self):
        super(MSA_1024, self).__init__()
        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv(32, 1, 1)
        self.conv2 = nn.Conv(16, 16, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(16)
        self.conv3 = nn.Conv(16, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm(16)
        self.conv4 = nn.Conv(32, 16, 1)
        self.bn4 = nn.BatchNorm(16)
        self.conv5 = nn.Conv(16, 16, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm(16)

    def execute(self, feat_mask, feat_struct):
        y = nn.relu(self.bn4(self.conv4(feat_mask)))
        x = self.act(self.conv1(feat_mask))
        x = (x * feat_struct)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3((x + feat_struct))))
        y = nn.relu(self.bn5(self.conv5((x + y))))
        return y


class UnionDe(nn.Module):

    def __init__(self):
        super(UnionDe, self).__init__()
        self.TSA_0 = TSA()
        self.TSA_1 = TSA()
        self.TSA_2 = TSA()
        self.MSA_3 = MSA_256()
        self.MSA_4 = MSA_512()
        self.MSA_5 = MSA_1024()
        self.conv_1 = nn.Sequential(nn.Conv(32, 32, 3, padding=1), nn.BatchNorm(32), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv(32, 32, 3, padding=1), nn.BatchNorm(32), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv(32, 32, 3, padding=1), nn.BatchNorm(32), nn.ReLU())
        self.conv_4 = nn.Sequential(nn.Conv(32, 32, 3, padding=1), nn.BatchNorm(32), nn.ReLU())
        self.conv_4_reduce = nn.Sequential(nn.Conv(32, 16, 3, padding=1), nn.BatchNorm(16), nn.ReLU())
        self.conv_5 = nn.Sequential(nn.Conv(16, 16, 3, padding=1), nn.BatchNorm(16), nn.ReLU())

    def execute(self, feat_trunk, feat_struct):
        mask = self.TSA_0(feat_trunk[0], feat_struct[0])
        temp = self.TSA_1(feat_trunk[1], feat_struct[1])
        maskup = nn.interpolate(mask, size=temp.shape[2:], mode='bilinear')
        temp = (maskup + temp)
        mask = self.conv_1(temp)
        temp = self.TSA_2(feat_trunk[2], feat_struct[2])
        maskup = nn.interpolate(mask, size=temp.shape[2:], mode='bilinear')
        temp = (maskup + temp)
        mask = self.conv_2(temp)
        maskup = nn.interpolate(mask, size=feat_struct[3].shape[2:], mode='bilinear')
        temp = self.MSA_3(maskup, feat_struct[3])
        temp = (maskup + temp)
        mask = self.conv_3(temp)
        maskup = nn.interpolate(mask, size=feat_struct[4].shape[2:], mode='bilinear')
        temp = self.MSA_4(maskup, feat_struct[4])
        temp = (maskup + temp)
        mask = self.conv_4(temp)
        maskup = nn.interpolate(mask, size=feat_struct[5].shape[2:], mode='bilinear')
        temp = self.MSA_5(maskup, feat_struct[5])
        maskup = self.conv_4_reduce(maskup)
        temp = (maskup + temp)
        mask = self.conv_5(temp)
        return mask
