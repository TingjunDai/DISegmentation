import os
import math


class Config():
    def __init__(self) -> None:
        self.optimizer = ['Adam', 'AdamW', 'SGD'][2]
        self.batch_size = 2
        self.lr = 0.05  # 1e-5 * math.sqrt(self.batch_size / 5)  # adapt the lr linearly
        self.lr_decay_epochs = [1e4]  # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5

        self.lambdas_cls = {
            'ce': 5.0
        }
        # Loss
        self.lambdas_gt_encoder = {
            # not 0 means opening this loss
            'bce': 1 * 1,  # high performance
            'iou': 0.5 * 0,  # 0 / 255
            'iou_patch': 0.5 * 0,  # 0 / 255, win_size = (64, 64)
            'mse': 150 * 0,  # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 10 * 0,  # help contours,
            'cnt': 5 * 0,  # help contours
        }
        self.lambdas_isnet_1 = {
            # not 0 means opening this loss
            'bce': 1 * 1,  # high performance
            'iou': 0.5 * 0,  # 0 / 255
            'iou_patch': 0.5 * 0,  # 0 / 255, win_size = (64, 64)
            'mse': 1 * 0,  # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 10 * 0,  # help contours,
            'cnt': 5 * 0,  # help contours
        }
        self.lambdas_isnet_2 = {
            # not 0 means opening this loss
            'bce': 1 * 0,  # high performance
            'iou': 0.5 * 0,  # 0 / 255
            'iou_patch': 0.5 * 0,  # 0 / 255, win_size = (64, 64)
            'mse': 1 * 1,  # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 10 * 0,  # help contours,
            'cnt': 5 * 0,  # help contours
        }
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,  # high performance
            'iou': 0.5 * 1,  # 0 / 255
            'iou_patch': 0.5 * 0,  # 0 / 255, win_size = (64, 64)
            'mse': 150 * 0,  # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 10 * 1,  # help contours,
            'cnt': 5 * 0,  # help contours
        }
        self.lambdas_udun = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 1 * 0,  # high performance
            'iou': 1 * 1,  # 0 / 255
            'iou_patch': 0.5 * 0,  # 0 / 255, win_size = (64, 64)
            'mse': 150 * 0,  # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 10 * 0,  # help contours,
            'cnt': 5 * 0,  # help contours
        }
        self.resnet50_weight = "saved_model/udun/resnet50-19c8e357.pth"
        self.udun_weight = "../saved_model/udun/udun-trained-R50.pth"
        self.birefnet_weight = "../saved_model/birefnet/BiRefNet_DIS_ep500-swin_v1_tiny.pth"
        self.birefnet_weight_2 = "../saved_model/birefnet/ep200.pth"
        self.isnet_weight = "../saved_model/isnet/isnet.pth"
        self.model = 'UDUN'
        self.data_root_dir = '../../datasets/dis'
        self.task = 'DIS5K'
        self.dataset = 'DIS5K'
        self.eval_model = 'birefnet'
        self.verbose_eval = True

        self.ms_supervision = True
        self.out_ref = self.ms_supervision and False
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.dec_blk = ['BasicDecBlk', 'ResBlk', 'HierarAttDecBlk'][0]
        self.lat_blk = ['BasicLatBlk'][0]
        self.mul_scl_ipt = ['', 'add', 'cat'][2]

        self.squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
        self.auxiliary_classification = False
        self.locate_head = False
        self.refine = ['', 'itself', 'RefUNet', 'Refiner', 'RefinerPVTInChannels4'][0]
        self.progressive_ref = self.refine and True
        self.ender = self.progressive_ref and False
        self.freeze_bb = False

        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',  # 0, 1, 2
            'pvt_v2_b2', 'pvt_v2_b5',  # 3-bs10, 4-bs5
            'swin_v1_b', 'swin_v1_l',  # 5-bs9, 6-bs6
            'swin_v1_t', 'swin_v1_s',  # 7, 8
            'pvt_v2_b0', 'pvt_v2_b1',  # 9, 10
        ][7]
        self.lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
            'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
            'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
        }[self.bb]

        self.dec_channels_inter = ['fixed', 'adap'][0]
        self.dec_att = ['', 'ASPP', 'ASPPDeformable'][1]

        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]

        self.cxt_num = [0, 3][1]  # multi-scale skip connections from encoder
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        self.weights = {
            'swin_v1_t': '../saved_model/birefnet/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',
        }

        run_sh_file = [f for f in os.listdir('.') if 'train.sh' == f] + [os.path.join('..', f) for f in os.listdir('..')
                                                                         if 'train.sh' == f]
        with open(run_sh_file[0], 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.save_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
