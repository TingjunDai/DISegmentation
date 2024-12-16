import jittor as jt
from jittor import init
from jittor import nn
from collections import OrderedDict
from jittor import nn
# from kornia.filters import laplacian
from config import Config
from dataset import class_labels_TR_sorted
from models.modules.decoder_blocks import BasicDecBlk, ResBlk, HierarAttDecBlk
from models.modules.lateral_blocks import BasicLatBlk
from models.backbones.build_backbone import build_backbone

jt.flags.use_cuda = 1

class BiRefNet(nn.Module):

    def __init__(self, bb_pretrained=True):
        super(BiRefNet, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.bb = self.bb = build_backbone(self.config.bb, pretrained=bb_pretrained)
        channels = self.config.lateral_channels_in_collection
        if self.config.auxiliary_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_head = nn.Sequential(nn.Linear(channels[0], len(class_labels_TR_sorted)))
        if self.config.squeeze_block:
            self.squeeze_module = nn.Sequential(
                *[eval(self.config.squeeze_block.split('_x')[0])((channels[0] + sum(self.config.cxt)), channels[0]) for
                  _ in range(eval(self.config.squeeze_block.split('_x')[1]))])
        self.decoder = Decoder(channels)
        if self.config.locate_head:
            self.locate_header = nn.ModuleList([BasicDecBlk(channels[0], channels[(- 1)]),
                                                nn.Sequential(nn.Conv(channels[(- 1)], 1, 1, stride=1, padding=0))])
        if self.config.ender:
            self.dec_end = nn.Sequential(nn.Conv(1, 16, 3, stride=1, padding=1), nn.Conv(16, 1, 3, stride=1, padding=1),
                                         nn.ReLU())
        if self.config.freeze_bb:
            print(self.named_parameters())
            for (key, value) in self.named_parameters():
                if ('bb.' in key) and ('refiner.' not in key):
                    value.requires_grad = False

    def forward_enc(self, x):
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            (x1, x2, x3, x4) = self.bb(x)
            if self.config.mul_scl_ipt == 'cat':
                (B, C, H, W) = x.shape
                (x1_, x2_, x3_, x4_) = self.bb(
                    nn.interpolate(x, size=((H // 2), (W // 2)), mode='bilinear', align_corners=True))
                x1 = jt.contrib.concat([x1, nn.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)],
                                       dim=1)
                x2 = jt.contrib.concat([x2, nn.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)],
                                       dim=1)
                x3 = jt.contrib.concat([x3, nn.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)],
                                       dim=1)
                x4 = jt.contrib.concat([x4, nn.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)],
                                       dim=1)
            elif self.config.mul_scl_ipt == 'add':
                (B, C, H, W) = x.shape
                (x1_, x2_, x3_, x4_) = self.bb(
                    nn.interpolate(x, size=((H // 2), (W // 2)), mode='bilinear', align_corners=True))
                x1 = (x1 + nn.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True))
                x2 = (x2 + nn.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True))
                x3 = (x3 + nn.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True))
                x4 = (x4 + nn.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True))
        class_preds = (self.cls_head(self.avgpool(x4).view((x4.shape[0], (- 1)))) if (
                    self.training and self.config.auxiliary_classification) else None)
        if self.config.cxt:
            x4 = jt.contrib.concat((*[nn.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                                      nn.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                                      nn.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True)][
                                     (- len(self.config.cxt)):], x4), dim=1)
        return (x1, x2, x3, x4), class_preds

    def forward_ori(self, x):
        ((x1, x2, x3, x4), class_preds) = self.forward_enc(x)
        if self.config.squeeze_block:
            x4 = self.squeeze_module(x4)
        features = [x, x1, x2, x3, x4]
        # if self.training and self.config.out_ref:
        #     features.append(laplacian(jt.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        scaled_preds = self.decoder(features)
        return scaled_preds, class_preds

    def execute(self, x):
        (scaled_preds, class_preds) = self.forward_ori(x)
        class_preds_lst = [class_preds]
        return [scaled_preds, class_preds_lst] if self.training else scaled_preds


class Decoder(nn.Module):

    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        DecoderBlock = eval(self.config.dec_blk)
        LateralBlock = eval(self.config.lat_blk)
        if self.config.dec_ipt:
            self.split = self.config.dec_ipt_split
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            self.ipt_blk4 = DBlock((((2 ** 8) * 3) if self.split else 3), [N_dec_ipt, (channels[0] // 8)][ipt_cha_opt],
                                   inter_channels=ic)
            self.ipt_blk3 = DBlock((((2 ** 6) * 3) if self.split else 3), [N_dec_ipt, (channels[1] // 8)][ipt_cha_opt],
                                   inter_channels=ic)
            self.ipt_blk2 = DBlock((((2 ** 4) * 3) if self.split else 3), [N_dec_ipt, (channels[2] // 8)][ipt_cha_opt],
                                   inter_channels=ic)
            self.ipt_blk1 = DBlock((((2 ** 0) * 3) if self.split else 3), [N_dec_ipt, (channels[3] // 8)][ipt_cha_opt],
                                   inter_channels=ic)
        else:
            self.split = None
        self.decoder_block4 = DecoderBlock(channels[0], channels[1])
        self.decoder_block3 = DecoderBlock(
            (channels[1] + ([N_dec_ipt, (channels[0] // 8)][ipt_cha_opt] if self.config.dec_ipt else 0)), channels[2])
        self.decoder_block2 = DecoderBlock(
            (channels[2] + ([N_dec_ipt, (channels[1] // 8)][ipt_cha_opt] if self.config.dec_ipt else 0)), channels[3])
        self.decoder_block1 = DecoderBlock(
            (channels[3] + ([N_dec_ipt, (channels[2] // 8)][ipt_cha_opt] if self.config.dec_ipt else 0)),
            (channels[3] // 2))
        self.conv_out1 = nn.Sequential(
            nn.Conv(((channels[3] // 2) + ([N_dec_ipt, (channels[3] // 8)][ipt_cha_opt] if self.config.dec_ipt else 0)),
                    1, 1, stride=1, padding=0))
        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])
        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv(channels[1], 1, 1, stride=1, padding=0)
            self.conv_ms_spvn_3 = nn.Conv(channels[2], 1, 1, stride=1, padding=0)
            self.conv_ms_spvn_2 = nn.Conv(channels[3], 1, 1, stride=1, padding=0)
            if self.config.out_ref:
                _N = 16
                self.gdt_convs_3 = nn.Sequential(nn.Conv(channels[2], _N, 3, stride=1, padding=1), nn.BatchNorm(_N),
                                                 nn.ReLU())
                self.gdt_convs_2 = nn.Sequential(nn.Conv(channels[3], _N, 3, stride=1, padding=1), nn.BatchNorm(_N),
                                                 nn.ReLU())
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv(_N, 1, 1, stride=1, padding=0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv(_N, 1, 1, stride=1, padding=0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv(_N, 1, 1, stride=1, padding=0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv(_N, 1, 1, stride=1, padding=0))

    def get_patches_batch(self, x, p):
        (_size_h, _size_w) = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = jt.split(x[idx], _size_w, -1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [p.unsqueeze(0) for p in jt.split(column_x, _size_h, -2)]
            patch_sample = jt.contrib.concat(patches_x, dim=1)
            patches_batch.append(patch_sample)
        return jt.contrib.concat(patches_batch, dim=0)

    def execute(self, features):
        if self.training and self.config.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            (x, x1, x2, x3, x4, gdt_gt) = features
        else:
            (x, x1, x2, x3, x4) = features
        outs = []
        p4 = self.decoder_block4(x4)
        m4 = (self.conv_ms_spvn_4(p4) if self.config.ms_supervision else None)
        _p4 = nn.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = (_p4 + self.lateral_block4(x3))
        if self.config.dec_ipt:
            patches_batch = (self.get_patches_batch(x, _p3) if self.split else x)
            _p3 = jt.contrib.concat((_p3, self.ipt_blk4(
                nn.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), dim=1)
        p3 = self.decoder_block3(_p3)
        m3 = (self.conv_ms_spvn_3(p3) if self.config.ms_supervision else None)
        if self.config.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                m3_dia = m3
                gdt_label_main_3 = (
                            gdt_gt * nn.interpolate(m3_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True))
                outs_gdt_label.append(gdt_label_main_3)
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            p3 = (p3 * gdt_attn_3)
        _p3 = nn.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = (_p3 + self.lateral_block3(x2))
        if self.config.dec_ipt:
            patches_batch = (self.get_patches_batch(x, _p2) if self.split else x)
            _p2 = jt.contrib.concat((_p2, self.ipt_blk3(
                nn.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), dim=1)
        p2 = self.decoder_block2(_p2)
        m2 = (self.conv_ms_spvn_2(p2) if self.config.ms_supervision else None)
        if self.config.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                m2_dia = m2
                gdt_label_main_2 = (
                            gdt_gt * nn.interpolate(m2_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True))
                outs_gdt_label.append(gdt_label_main_2)
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            p2 = (p2 * gdt_attn_2)
        _p2 = nn.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = (_p2 + self.lateral_block2(x1))
        if self.config.dec_ipt:
            patches_batch = (self.get_patches_batch(x, _p1) if self.split else x)
            _p1 = jt.contrib.concat((_p1, self.ipt_blk2(
                nn.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), dim=1)
        _p1 = self.decoder_block1(_p1)
        _p1 = nn.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        if self.config.dec_ipt:
            patches_batch = (self.get_patches_batch(x, _p1) if self.split else x)
            _p1 = jt.contrib.concat((_p1, self.ipt_blk1(
                nn.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=True))), dim=1)
        p1_out = self.conv_out1(_p1)
        if self.config.ms_supervision:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        return outs if (not (self.config.out_ref and self.training)) else ([outs_gdt_pred, outs_gdt_label], outs)


class SimpleConvs(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, inter_channels=64) -> None:
        super().__init__()
        self.conv1 = nn.Conv(in_channels, inter_channels, 3, stride=1, padding=1)
        self.conv_out = nn.Conv(inter_channels, out_channels, 3, stride=1, padding=1)

    def execute(self, x):
        return self.conv_out(self.conv1(x))
