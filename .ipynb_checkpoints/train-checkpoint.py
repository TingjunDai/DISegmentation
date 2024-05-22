import os
import time
import jittor as jt
from jittor import init
from jittor import nn
from models.isnet_gt import ISNetGTEncoder
from models.isnet import ISNetDIS
from dataset import get_gt_loader, get_train_loader
from loss import PixLoss, ClsLoss
from utils import AverageMeter, Logger
import argparse

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--trainset', default='DIS5K', type=str, help="Options: 'DIS5K'")
parser.add_argument('--ckpt_dir', default=None, help='Temporary folder')
parser.add_argument('--testsets', default='DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4', type=str)
parser.add_argument('--trainer', default='GTEncoder', type=str, help='Current Pretrain Model')
parser.add_argument('--gt_encoder_pth', default='', type=str, help='Well Trained GT Encoder Path')
args = parser.parse_args()

logger = Logger(os.path.join("", "log.txt"))
logger_loss_idx = 1


class ISNet_GTEncoder_Trainer:
    def __init__(
            self, data_loaders, model_opt,
    ):
        self.model, self.optimizer = model_opt
        self.train_loader = data_loaders

        # Setting Losses
        self.pix_loss = PixLoss('ISNet-GT')

        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        inputs = batch[0]
        gts = batch[1]
        gt_preds, _ = self.model(inputs)
        # Loss
        loss = self.pix_loss(gt_preds, jt.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss'] = loss.item()

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            self._train_batch(batch)
            # Logger
            # if batch_idx % 20 == 0:
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx,
                                                                   self.train_loader.total_len / 2)
            info_loss = 'Training Losses'
            for loss_name, loss_value in self.loss_dict.items():
                info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
            logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs,
                                                                                        loss=self.loss_log)
        logger.info(info_loss)

        return self.loss_log.avg


class ISNet_Trainer:
    def __init__(
            self, data_loaders, model_opt
    ):
        self.model, self.gt_model, self.optimizer = model_opt
        self.train_loader = data_loaders

        # Setting Losses
        self.pix_loss_1 = PixLoss('ISNet-GT_1')
        self.pix_loss_2 = PixLoss('ISNet-GT_2')  # 有问题，loss函数实现不适配
        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        inputs = batch[0]
        gts = batch[1]
        gt_preds, is_preds = self.model(inputs)
        _, is_gt = self.gt_model(gts)
        # Loss
        loss = self.pix_loss_1(gt_preds, jt.clamp(gts, 0, 1)) * 1.0 + self.pix_loss_2(is_preds, is_gt)  # 有问题，得改
        self.loss_dict['loss'] = loss.item()

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            self._train_batch(batch)
            # Logger
            # if batch_idx % 20 == 0:
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx,
                                                                   len(self.train_loader))
            info_loss = 'Training Losses'
            for loss_name, loss_value in self.loss_dict.items():
                info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
            logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs,
                                                                                        loss=self.loss_log)
        logger.info(info_loss)

        return self.loss_log.avg


def main():
    if args.trainer == 'GTEncoder':
        model = ISNetGTEncoder()
        trainer = ISNet_GTEncoder_Trainer(
            data_loaders=get_gt_loader(gt_root='../DIS5K/DIS-TR/gt/', batchsize=2, trainsize=1024),
            model_opt=(model, jt.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0))
        )
    else:
        gt_model = ISNetGTEncoder()
        gt_model.load_state_dict(jt.load(args.gt_encoder_pth))
        model = ISNetDIS()
        trainer = ISNet_GTEncoder_Trainer(
            data_loaders=get_train_loader(image_root='../DIS5K/DIS-TR/im/', gt_root=('../DIS5K/DIS-TR/gt/'),
                                          batchsize=1, trainsize=1024),
            model_opt=(
            model, gt_model, jt.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0))
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        # Save checkpoint
        # DDP
        jt.save(trainer.model.state_dict(), os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch)))


if __name__ == '__main__':
    main()
