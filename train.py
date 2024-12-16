import os
import time
import jittor as jt
from jittor import init
from jittor import nn
from models.isnet.isnet_gt import ISNetGTEncoder
from models.isnet.isnet import ISNet
from models.birefnet.birefnet import BiRefNet
from models.udun.udun import UDUN
from models.mvanet.mvanet import MVANet
from dataset import get_data_loader
from loss import PixLoss, ClsLoss, SupLoss, IoULoss
from utils import AverageMeter, Logger, split_head_and_base
import argparse
from config import Config
import cv2
from evaluation.metrics import Fmeasure
import numpy as np

config = Config()
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Logger folder')
args = parser.parse_args()

# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))


def init_models_optimizers(epochs):
    model = []
    optimizer = []
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True)
    elif config.model == 'UDUN':
        model = UDUN(bb_pretrained=True)
    elif config.model == 'ISNet':
        model = ISNet()
    elif config.model == 'ISNet_GTEncoder':
        model = ISNetGTEncoder()
    elif config.model == 'MVANet':
        model = MVANet(bb_pretrained=True)
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
        self.train_loader, self.valid_loader = data_loaders

        # Setting Losses
        self.pix_loss = nn.BCELoss()

        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        inputs = batch[1]
        gts = batch[1]
        gt_preds, _ = self.model(inputs)
        # Loss
        loss = self.pix_loss(gt_preds, jt.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss'] = loss.item()

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)
        
    def valid_gt_encoder(self):
        self.model.eval()
        fmeasure_calculator = Fmeasure(beta=0.3)
        for batch_idx, batch in enumerate(self.valid_loader):
            inputs = batch[1]
            gts = batch[1]
            label_paths = batch[-1]
            with jt.no_grad():
                scaled_preds = self.model(inputs)[0][0].sigmoid()
            for idx_sample in range(scaled_preds.shape[0]):
                res = nn.interpolate(
                    scaled_preds[idx_sample].unsqueeze(0),
                    size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                    mode='bilinear',
                    align_corners=True
                )
                gt = nn.interpolate(
                    gts[idx_sample].unsqueeze(0),
                    size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                    mode='bilinear',
                    align_corners=True
                )
                fmeasure_calculator.step(pred=res.squeeze(0).squeeze(0).numpy() * 255., gt=gt.squeeze(0).squeeze(0).numpy() * 255.)
            if batch_idx > len(self.valid_loader) // 5:
                break
        self.model.train()
        return fmeasure_calculator.get_results()['fm']['curve'].max()

    def train_epoch(self, epoch):
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader) // config.batch_size)
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg


class ISNet_Trainer:
    def __init__(
            self, data_loaders, model_opt_lrsch, gt_model
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader, self.valid_loader = data_loaders
        self.gt_model = gt_model

        # Setting Losses
        self.bce_loss = PixLoss('ISNet')
        if config.interm_sup:
            self.sup_loss = SupLoss()
        # Others
        self.loss_log = AverageMeter()
        
    def valid(self):
        self.model.eval()
        fmeasure_calculator = Fmeasure(beta=0.3)
        for batch_idx, batch in enumerate(self.valid_loader):
            inputs = batch[0]
            gts = batch[1]
            label_paths = batch[-1]
            with jt.no_grad():
                scaled_preds = self.model(inputs)[0][0].sigmoid()
            for idx_sample in range(scaled_preds.shape[0]):
                res = nn.interpolate(
                    scaled_preds[idx_sample].unsqueeze(0),
                    size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                    mode='bilinear',
                    align_corners=True
                )
                gt = nn.interpolate(
                    gts[idx_sample].unsqueeze(0),
                    size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                    mode='bilinear',
                    align_corners=True
                )
                fmeasure_calculator.step(pred=res.squeeze(0).squeeze(0).numpy() * 255., gt=gt.squeeze(0).squeeze(0).numpy() * 255.)
        self.model.train()
        return fmeasure_calculator.get_results()['fm']['curve'].max()

    def _train_batch(self, batch):
        inputs = batch[0]
        gts = batch[1]
        gt_preds, is_preds = self.model(inputs)
        if config.interm_sup:
            with jt.no_grad():
                _, is_gt = self.gt_model(gts)
        # Loss
        loss = (self.bce_loss(gt_preds, jt.clamp(gts, 0, 1)) + self.sup_loss(is_preds, is_gt)) if config.interm_sup else self.bce_loss(gt_preds, jt.clamp(gts, 0, 1))
        self.loss_dict['loss'] = loss.item()

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)

    def train_epoch(self, epoch):
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader) // config.batch_size)
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg


class UDUN_Trainer:
    def __init__(
            self, data_loaders, model_opt_lrsch
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = data_loaders

        # Setting Losses
        self.iou_loss = IoULoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss() 
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
        loss_t = self.bce_logits_loss(out_trunk, trunk)
        struct = nn.interpolate(structs, size=out_struct.size()[2:], mode='bilinear')
        loss_s = self.bce_logits_loss(out_struct, struct)
        mask = nn.interpolate(gts, size=out_mask.size()[2:], mode='bilinear')
        lossmask = self.bce_logits_loss(out_mask, mask) + self.iou_loss(out_mask.sigmoid(), mask.sigmoid())
        loss = (loss_t + loss_s + lossmask) / 2

        self.loss_dict['loss'] = loss.item()

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)

    def train_epoch(self, epoch):
        self.optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (args.epochs + 1) * 2 - 1)) * config.lr * 0.1
        self.optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (args.epochs + 1) * 2 - 1)) * config.lr
        
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader) // config.batch_size)
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
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
        
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

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
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
        # Loss
        loss_pix = self.pix_loss(scaled_preds, jt.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0
        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)

    def train_epoch(self, epoch):
        self.model.train()
        self.loss_dict = {}
        if epoch > args.epochs + config.finetune_last_epochs:
            if config.task == 'Matting':
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9
        
        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader) // config.batch_size)
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg
    
    
class MVANet_Trainer:
    def __init__(
            self, data_loaders, model_opt_lrsch
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = data_loaders
        self.pix_loss = PixLoss('MVANet')
        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        inputs = batch[0]
        gts = batch[1]
        sideout5, sideout4, sideout3, sideout2, sideout1, final, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3,tokenattmap2,tokenattmap1 = self.model(inputs)
        # Loss
        b, c, h, w = gts.size()
        target_1 = nn.interpolate(gts, size=h // 4, mode='nearest')
        target_2 = nn.interpolate(gts, size=h // 8, mode='nearest')
        target_3 = nn.interpolate(gts, size=h // 16, mode='nearest')
        target_4 = nn.interpolate(gts, size=h // 32, mode='nearest')
        target_5 = nn.interpolate(gts, size=h // 64, mode='nearest')
        loss1 = self.pix_loss(sideout5.unsqueeze(0), target_4)
        loss2 = self.pix_loss(sideout4.unsqueeze(0), target_3)
        loss3 = self.pix_loss(sideout3.unsqueeze(0), target_2)
        loss4 = self.pix_loss(sideout2.unsqueeze(0), target_1)
        loss5 = self.pix_loss(sideout1.unsqueeze(0), target_1)
        loss6 = self.pix_loss(final.unsqueeze(0), gts)
        loss7 = self.pix_loss(glb5.unsqueeze(0), target_5)
        loss8 = self.pix_loss(glb4.unsqueeze(0), target_4)
        loss9 = self.pix_loss(glb3.unsqueeze(0), target_3)
        loss10 = self.pix_loss(glb2.unsqueeze(0), target_2)
        loss11 = self.pix_loss(glb1.unsqueeze(0), target_2)
        loss12 = self.pix_loss(tokenattmap4.unsqueeze(0), target_3)
        loss13 = self.pix_loss(tokenattmap3.unsqueeze(0), target_2)
        loss14 = self.pix_loss(tokenattmap2.unsqueeze(0), target_1)
        loss15 = self.pix_loss(tokenattmap1.unsqueeze(0), target_1)
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.3*(loss7 + loss8 + loss9 + loss10 + loss11)+ 0.3*(loss12 + loss13 + loss14 + loss15)
                

        self.loss_dict['loss'] = loss.item()

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.step(loss)

    def train_epoch(self, epoch):  
        decay = config.decay_rate * (epoch // config.decay_epochs)
        for param_group in self.optimizer.param_groups:
            if 'lr' in param_group:
                param_group['lr'] *= decay
        
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader) // config.batch_size)
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg


def main():
    trainer = []
    if config.model == 'ISNet_GTEncoder':
        trainer = ISNet_GTEncoder_Trainer(
            data_loaders=(get_data_loader(datasets=config.training_set, batch_size=config.batch_size),
                          get_data_loader(datasets=config.training_set, batch_size=config.batch_size, is_train=False)),
            model_opt_lrsch=init_models_optimizers(args.epochs)
        )
        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(epoch)
            if trainer.valid_gt_encoder() > 0.99:
                jt.save(
                    trainer.model.state_dict(),
                    os.path.join(args.ckpt_dir, 'ep{}.pkl'.format(epoch))
                )
                break
    elif config.model == 'ISNet':
        gt_model = None
        if config.interm_sup:
            gt_model = ISNetGTEncoder()
            gt_model.load_state_dict(jt.load(config.backbone_weights['gt_encoder']))
            gt_model.eval()
        trainer = ISNet_Trainer(
            data_loaders=(get_data_loader(datasets=config.training_set, batch_size=config.batch_size),
                          get_data_loader(datasets=config.training_set, batch_size=config.batch_size, is_train=False)),
            model_opt_lrsch=init_models_optimizers(args.epochs),
            gt_model=gt_model
        )
        last_f1 = 0
        notgood_cnt = 0
        for epoch in range(1, args.epochs + 1):
            notgood_cnt += 1
            train_loss = trainer.train_epoch(epoch)
            tmp_f1 = trainer.valid()
            if  tmp_f1 > last_f1:
                jt.save(
                    trainer.model.state_dict(),
                    os.path.join(args.ckpt_dir, 'ep{}.pkl'.format(epoch))
                )
                last_f1 = tmp_f1
                notgood_cnt = 0
            if notgood_cnt >= config.early_stop:
                print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
                break
    elif config.model == 'UDUN':
        trainer = UDUN_Trainer(
            data_loaders=get_data_loader(datasets=config.training_set, batch_size=config.batch_size),
            model_opt_lrsch=init_models_optimizers(args.epochs)
        )
        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(epoch)
            # Save checkpoint
            if epoch >= args.epochs * config.save_ratio:
                jt.save(
                    trainer.model.state_dict(),
                    os.path.join(args.ckpt_dir, 'ep{}.pkl'.format(epoch))
                )
    elif config.model == 'BiRefNet':
        trainer = BiRefNet_Trainer(
            data_loaders=get_data_loader(datasets=config.training_set, batch_size=config.batch_size),
            model_opt_lrsch=init_models_optimizers(args.epochs)
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(epoch)
            # Save checkpoint
            if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
                jt.save(
                    trainer.model.state_dict(),
                    os.path.join(args.ckpt_dir, 'ep{}.pkl'.format(epoch))
                )
    elif config.model == 'MVANet':
        trainer = MVANet_Trainer(
            data_loaders=get_data_loader(datasets=config.training_set, batch_size=1),
            model_opt_lrsch=init_models_optimizers(args.epochs)
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(epoch)
            # Save checkpoint
            if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
                jt.save(
                    trainer.model.state_dict(),
                    os.path.join(args.ckpt_dir, 'ep{}.pkl'.format(epoch))
                )


if __name__ == '__main__':
    main()
