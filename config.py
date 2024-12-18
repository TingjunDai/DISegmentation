import os
import math


class Config():
    def __init__(self) -> None:
        #### BASIC settings  ####
        # PATH settings
        # Make up your file system as: SYS_HOME_DIR/codes, SYS_HOME_DIR/datasets/dis/xx, SYS_HOME_DIR/saved_model/xx
        self.sys_home_dir = [os.path.expanduser('~'), '/maqi/DTJImPart'][1]
        self.data_root_dir = os.path.join(self.sys_home_dir, 'datasets')
        # self.weights_root_dir = os.path.join(self.sys_home_dir, 'saved_model')

        # datasets settings
        self.task = ['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'][0]
        self.testsets = {
            # Benchmarks
            'DIS5K': ','.join(['DIS-VD', 'DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'][:1]),
            'COD': ','.join(['CHAMELEON', 'NC4K', 'TE-CAMO', 'TE-COD10K']),
            'HRSOD': ','.join(['DAVIS-S', 'TE-HRSOD', 'TE-UHRSD', 'DUT-OMRON', 'TE-DUTS']),
            # Practical use
            'General': ','.join(['DIS-VD', 'TE-P3M-500-NP']),
            'General-2K': ','.join(['DIS-VD', 'TE-P3M-500-NP']),
            'Matting': ','.join(['TE-P3M-500-NP', 'TE-AM-2k']),
        }[self.task]
        datasets_all = '+'.join([ds for ds in (os.listdir(os.path.join(self.data_root_dir, self.task)) if os.path.isdir(os.path.join(self.data_root_dir, self.task)) else []) if ds not in self.testsets.split(',')])
        self.training_set = {
            'DIS5K': ['DIS-TR', 'DIS-TR+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'][0],
            'COD': 'TR-COD10K+TR-CAMO',
            'HRSOD': ['TR-DUTS', 'TR-HRSOD', 'TR-UHRSD', 'TR-DUTS+TR-HRSOD', 'TR-DUTS+TR-UHRSD', 'TR-HRSOD+TR-UHRSD', 'TR-DUTS+TR-HRSOD+TR-UHRSD'][5],
            'General': datasets_all,
            'General-2K': datasets_all,
            'Matting': datasets_all,
        }[self.task]
        
        # MODEL settings
        self.model = ['ISNet', 'UDUN', 'BiRefNet', 'ISNet_GTEncoder', 'MVANet'][1]
        
        # TRAIN settings
        self.batch_size = 1  # MVANet' batch size can only be 1
        self.optimizer = {
            'BiRefNet': 'AdamW',
            'ISNet_GTEncoder': 'Adam',
            'UDUN': 'SGD',
            'ISNet': 'Adam',
            'MVANet': 'Adam',
        }[self.model]
        self.lr = {
            'BiRefNet': (1e-4 if 'DIS5K' in self.task else 1e-5) * math.sqrt(self.batch_size / 4),
            'ISNet_GTEncoder': 1e-3,
            'UDUN': 0.05,
            'ISNet': 1e-3,
            'MVANet': 1e-5,
        }[self.model] # learning rate
        self.lr_decay_epochs = [1e4]  # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5
        self.size = (1024, 1024)  # input size
        self.preproc_methods = ['enhance', 'rotate', 'pepper', 'flip', 'crop'][:3] # data enhance method
        self.load_all = False  # Turn it on/off by your case. It may consume a lot of CPU memory but accelerate train by loading whole dataset at once
        
        self.backbone_weights = { # dir you download the backbone_weights
            'swin_v1_t': '../backbone_weights/birefnet/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',
            'swin_v1_b': '../backbone_weights/mvanet/swin_base_patch4_window12_384_22kto1k.pth',
            'resnet50': '../backbone_weights/udun/resnet50.pkl',
            'gt_encoder': '../backbone_weights/isnet/ep2.pkl'
        }
        # EVAL settings
        self.batch_size_valid = 1
        self.only_S_MAE = False # Turn it on for only evaluating Smeausre and MAE
        
        ####  Settings that vary depending on the model  ####
        # BiRefNet settings
        self.save_last = 20 
        self.save_step = 5 # save the weights starting from the last 20 epochs, and save every 5 epochs
        self.lambdas_birefnet = { # 10 types loss function
                # not 0 means opening this loss
                'bce': 30 * 1,          
                'bce_logits': 1 * 0,    # bce with sigmoid
                'iou': 0.5 * 1,         
                'iou_patch': 0.5 * 0,   # win_size = (64, 64)
                'mae': 30 * 0,
                'mse': 30 * 0,         # can smooth the saliency map
                'reg': 100 * 0,
                'ssim': 10 * 1,        # help contours
                'cnt': 5 * 0,          # help contours
                'structure': 5 * 0,    # structure loss from MVANet. wbce + wiou
        }
        self.lambdas_cls = {
            'ce': 5.0
        }
        self.finetune_last_epochs = [
            0,
            {
                'DIS5K': -20,
                'COD': -20,
                'HRSOD': -20,
                'General': -40,
                'General-2K': -20,
                'Matting': -20,
            }[self.task]
        ][1]    # choose 0 to skip
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
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
        
        self.birefnet_bb = [ # Backbones supported by BiRefNet
            'pvt_v2_b2', 'pvt_v2_b5',  # 0-bs10, 1-bs5
            'swin_v1_b', 'swin_v1_l',  # 2-bs9, 3-bs6
            'swin_v1_t', 'swin_v1_s',  # 4, 5
            'pvt_v2_b0', 'pvt_v2_b1',  # 6, 7
        ][4]
        self.lateral_channels_in_collection = {
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
            'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
            'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
        }[self.bb]
        
        self.dec_channels_inter = ['fixed', 'adap'][0]
        self.dec_att = ['', 'ASPP', 'ASPPDeformable'][1]  # ASPPDeformable has some error in module dcnv2

        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]

        self.cxt_num = [0, 3][1]  # multi-scale skip connections from encoder
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []
        # UDUN settings
        self.bu = 'resnet50' # Backbones supported by UDUN
        self.save_ratio = 0.75 # save the last 25% epochs
        # ISNet settings
        self.lambdas_isnet = {
            'bce': 1 * 1,
        }
        self.early_stop = 5  # if the model's performance does not improve within 5 epochs, terminate it
        self.interm_sup = True # Trun on to activate intermediate feature supervision 
        # MVANet settings
        self.mva_bb = [ # Backbones supported by MVANet
            'swin_v1_b', 'swin_v1_l',  # 0, 1
            'swin_v1_t', 'swin_v1_s',  # 2, 3
        ][0]
        self.mva_lateral_channels_in_collection = {
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
            'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
        }[self.mva_bb]
        self.lambdas_mvanet = {
            'structure': 1 * 1
        }
        self.decay_epochs = 60
        self.decay_rate = 0.9 # After 60 epochs, the learning rate starts to dacay by multiplying it by 0.9 each epoch 


# Return task for choosing settings in shell scripts.
if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(description='Only choose one argument to activate.')
    parser.add_argument('--print_task', action='store_true', help='print task name')
    parser.add_argument('--print_testsets', action='store_true', help='print validation set')
    args = parser.parse_args()

    config = Config()
    for arg_name, arg_value in args._get_kwargs():
        if arg_value:
            print(config.__getattribute__(arg_name[len('print_'):]))