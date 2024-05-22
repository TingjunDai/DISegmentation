import os
import time
import jittor as jt
from jittor import init
from jittor import nn
from models.isnet_gt import ISNetGTEncoder
from models.isnet import ISNetDIS
from models.birefnet import BiRefNet
from models.udun import UDUN
from dataset import get_gt_loader, get_train_loader, get_birefnet_loader, get_udun_loader
from loss import PixLoss, ClsLoss, SupLoss, IoULoss
from utils import AverageMeter, Logger, split_head_and_base
import argparse
from config import Config

config = Config()
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--ckpt_dir', default='../saved_model/gt_encoder', help='Logger folder')
parser.add_argument('--gt_encoder_pth', default='../saved_model/gt_encoder/ep1.pkl', type=str,
                    help='Well Trained GT Encoder Path')
args = parser.parse_args()

logger = Logger(os.path.join("", "log.txt"))
logger_loss_idx = 1


def init_models_optimizers(epochs):
    model = []
    optimizer = []
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True)
    elif config.model == 'UDUN':
        model = UDUN()
    elif config.model == 'ISNET':
        model = ISNetDIS()
    elif config.model == 'GTEncoder':
        model = ISNetGTEncoder()
    if config.optimizer == 'AdamW':
        optimizer = jt.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = jt.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    elif config.optimizer == 'SGD':
        if config.model == 'UDUN':
            base, head = split_head_and_base(model)
            optimizer = jt.optim.SGD([{'params': base}, {'params': head}], lr=config.lr, weight_decay=5e-4,
                                     momentum=0.9, nesterov=True)
        else:
            optimizer = jt.optim.SGD(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = jt.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    logger.info("Optimizer details:")
    logger.info(optimizer)
    logger.info("Scheduler details:")
    logger.info(lr_scheduler)

    return model, optimizer, lr_scheduler


class ISNet_GTEncoder_Trainer:
    def __init__(
            self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
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
            self, data_loaders, model_opt_lrsch
    ):
        (self.model, self.optimizer, self.lr_scheduler), self.gt_model = model_opt_lrsch
        self.train_loader = data_loaders

        # Setting Losses
        self.pix_loss_1 = PixLoss('ISNet_1')
        self.pix_loss_2 = SupLoss()
        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        inputs = batch[0]
        gts = batch[1]
        gt_preds, is_preds = self.model(inputs)
        _, is_gt = self.gt_model(gts)
        # Loss
        loss = self.pix_loss_1(gt_preds, jt.clamp(gts, 0, 1)) + self.pix_loss_2(is_preds, is_gt)
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


class UDUN_Trainer:
    def __init__(
            self, data_loaders, model_opt_lrsch
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = data_loaders

        # Setting Losses
        self.pix_loss = IoULoss()
        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        inputs = batch[0]
        gts = batch[1]
        trunks = batch[2]
        structs = batch[3]
        out_trunk, out_struct, out_mask = self.model(inputs)
        # Loss
        trunk = nn.interpolate(trunks, size=out_trunk.size()[2:], mode='bilinear')
        loss_t = nn.binary_cross_entropy_with_logits(out_trunk, trunk)
        mask = nn.interpolate(gts, size=out_mask.size()[2:], mode='bilinear')
        lossmask = nn.binary_cross_entropy_with_logits(out_mask, mask) + self.pix_loss(out_mask, mask)
        struct = nn.interpolate(structs, size=out_struct.size()[2:], mode='bilinear')
        loss_s = nn.binary_cross_entropy_with_logits(out_struct, struct)
        loss = (loss_t + loss_s + lossmask) / 2

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


class BiRefNet_Trainer:
    def __init__(
            self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = data_loaders

        # Setting Losses
        self.pix_loss = PixLoss('BiRefNet')
        self.cls_loss = ClsLoss()

        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        inputs = batch[0]
        gts = batch[1]
        class_labels = batch[2]
        scaled_preds, class_preds_lst = self.model(inputs)
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels) * 1.0
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix = self.pix_loss(scaled_preds, jt.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx,
                                                                       len(self.train_loader))
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs,
                                                                                        loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg


def main():
    trainer = []
    if config.model == 'GTEncoder':
        trainer = ISNet_GTEncoder_Trainer(
            data_loaders=get_gt_loader(gt_root='../../DIS5K/DIS5K/DIS-TR/gt/', batchsize=2, trainsize=1024),
            model_opt_lrsch=init_models_optimizers(args.epochs)
        )
    elif config.model == 'ISNET':
        gt_model = ISNetGTEncoder()
        gt_model.load_state_dict(jt.load(args.gt_encoder_pth))
        trainer = ISNet_Trainer(
            data_loaders=get_train_loader(image_root='../../DIS5K/DIS5K/DIS-TR/im/', gt_root='../../DIS5K/DIS5K/DIS'
                                                                                             '-TR/gt/',
                                          batchsize=1, trainsize=1024),
            model_opt_lrsch=(init_models_optimizers(args.epochs), gt_model)
        )
    elif config.model == 'UDUN':
        trainer = UDUN_Trainer(
            data_loaders=get_udun_loader(image_root='../../DIS5K/DIS5K/DIS-TR/im/',
                                         gt_root='../../DIS5K/DIS5K/DIS-TR/gt/',
                                         trunk_root='../../DIS5K/DIS5K/DIS-TR/trunk-origin/',
                                         struct_root='../../DIS5K/DIS5K/DIS-TR/struct-origin/',
                                         batchsize=1, trainsize=1024),
            model_opt_lrsch=init_models_optimizers(args.epochs)
        )
    elif config.model == 'BiRefNet':
        trainer = BiRefNet_Trainer(
            data_loaders=get_birefnet_loader(image_root='../../datasets/dis/DIS5K/DIS-TR/im/',
                                             gt_root='../../datasets/dis/DIS5K/DIS-TR/gt/',
                                             batchsize=config.batch_size, trainsize=1024),
            model_opt_lrsch=init_models_optimizers(args.epochs)
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        # Save checkpoint
        # DDP
        if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
            jt.save(
                trainer.model.state_dict(),
                os.path.join(args.ckpt_dir, 'ep{}.pkl'.format(epoch))
            )


if __name__ == '__main__':
    main()
