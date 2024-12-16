import jittor as jt
from jittor import init
from jittor import nn

from models.modules.DCM import DCM
from models.backbones.resnet50 import resnet50
from models.modules.StrucDe import StrucDe
from models.modules.TrunkDe import TrunkDe
from models.modules.UnionDe import UnionDe
from config import Config

config = Config()

jt.flags.use_cuda = 1

class UDUN(nn.Module):
    def __init__(self):
        super(UDUN, self).__init__()

        self.convHR0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1), nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(16), nn.ReLU())
        self.convHR1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU())
        self.convHR2 = nn.Sequential(nn.Conv2d(256, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU())

        self.convHR3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU())
        self.convHR4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU())
        self.convHR5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU())

        self.convLR1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU())
        self.convLR2 = nn.Sequential(nn.Conv2d(256, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU())
        self.convLR3 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU())

        self.convLR4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU())
        self.convLR5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU())

        self.dcm = DCM()
        self.trunk = TrunkDe()
        self.struct = StrucDe()
        self.union_de = UnionDe()

        self.linear_t = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear_s = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16),
                                    nn.ReLU(), nn.Conv2d(16, 1, kernel_size=3, padding=1))

        self.bkbone = resnet50()
        self.bkbone.load_state_dict(jt.load('../saved_model/udun/resnet50.pkl'))

        if config.udun_weight != '':
            print('load checkpoint')
            self.load_state_dict(jt.load(config.udun_weight))

    def execute(self, x):
        y = nn.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        outHR0 = self.convHR0(x)

        outHR1, outHR2, outHR3, outHR4, outHR5 = self.bkbone(x)
        outLR1, outLR2, outLR3, outLR4, outLR5 = self.bkbone(y)

        outHR1, outHR2, outHR3, outHR4, outHR5 = self.convHR1(outHR1), self.convHR2(outHR2), self.convHR3(
            outHR3), self.convHR4(outHR4), self.convHR5(outHR5)
        outLR1, outLR2, outLR3, outLR4, outLR5 = self.convLR1(outLR1), self.convLR2(outLR2), self.convLR3(
            outLR3), self.convLR4(outLR4), self.convLR5(outLR5)

        out_T32, out_T43, out_T54 = self.trunk([outLR5, outLR4, outHR5, outHR4, outHR3])
        outLR3, outLR2, outLR1 = self.dcm([out_T32, out_T43, out_T54], [outLR3, outLR2, outLR1, outHR2])
        out_S1, out_S2, out_S3, out_S4, out_S5, out_S6 = self.struct([outLR3, outLR2, outLR1, outHR2, outHR1, outHR0])
        maskFeature = self.union_de([out_T32, out_T43, out_T54], [out_S1, out_S2, out_S3, out_S4, out_S5, out_S6])

        out_mask = self.linear(maskFeature)
        out_trunk = self.linear_t(out_T54)
        out_struct = self.linear_s(out_S6)

        return out_trunk, out_struct, out_mask
