import jittor as jt
from jittor import init
from jittor import nn

jt.flags.use_cuda = 1

class DCM(nn.Module):

    def __init__(self):
        super(DCM, self).__init__()
        self.convLR0 = nn.Conv(64, 32, 1)
        self.bnLR0 = nn.BatchNorm(32)
        self.convLR1 = nn.Conv(64, 32, 1)
        self.bnLR1 = nn.BatchNorm(32)
        self.convLR2 = nn.Conv(64, 32, 1)
        self.bnLR2 = nn.BatchNorm(32)
        self.convLR3 = nn.Conv(32, 32, 1)
        self.bnLR3 = nn.BatchNorm(32)

    def execute(self, featLR, featHR):
        temp = nn.relu(self.bnLR0(self.convLR0(featLR[0])))
        featHR0 = (featHR[0] - temp)
        temp = nn.relu(self.bnLR1(self.convLR1(featLR[1])))
        featHR1 = (featHR[1] - temp)
        temp = nn.relu(self.bnLR2(self.convLR2(featLR[2])))
        featHR2 = (featHR[2] - temp)
        return featHR0, featHR1, featHR2
