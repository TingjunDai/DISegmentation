import jittor as jt
from jittor import nn
from config import Config

import numpy as np


class ContourLoss(nn.Module):
    def __init__(self):
        super(ContourLoss, self).__init__()

    def execute(self, pred, target, weight=10):
        """
        target, pred: tensor of shape (B, C, H, W), where target[:,:,region_in_contour] == 1,
                        target[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        """
        # length term
        delta_r = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = jt.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        length = jt.mean(jt.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

        c_in = jt.ones_like(pred)
        c_out = jt.zeros_like(pred)

        region_in = jt.mean(pred * (target - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
        region_out = jt.mean((1 - pred) * (target - c_out) ** 2)
        region = region_in + region_out

        loss = weight * length + region

        return loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def execute(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = jt.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = jt.sum(target[i, :, :, :]) + jt.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)
        # return IoU/b
        return IoU


class PatchIoULoss(nn.Module):
    def __init__(self):
        super(PatchIoULoss, self).__init__()
        self.iou_loss = IoULoss()

    def execute(self, pred, target):
        win_y, win_x = 64, 64
        iou_loss = 0.
        for anchor_y in range(0, target.shape[0], win_y):
            for anchor_x in range(0, target.shape[1], win_y):
                patch_pred = pred[:, :, anchor_y:anchor_y + win_y, anchor_x:anchor_x + win_x]
                patch_target = target[:, :, anchor_y:anchor_y + win_y, anchor_x:anchor_x + win_x]
                patch_iou_loss = self.iou_loss(patch_pred, patch_target)
                iou_loss += patch_iou_loss
        return iou_loss


class ThrReg_loss(nn.Module):
    def __init__(self):
        super(ThrReg_loss, self).__init__()

    def execute(self, pred, gt=None):
        return jt.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))


class ClsLoss(nn.Module):
    """
    Auxiliary classification loss for each refined class output.
    """

    def __init__(self):
        super(ClsLoss, self).__init__()
        self.config = Config()
        self.lambdas_cls = self.config.lambdas_cls

        self.criterions_last = {
            'ce': nn.CrossEntropyLoss()
        }

    def execute(self, preds, gt):
        loss = 0.
        for _, pred_lvl in enumerate(preds):
            if pred_lvl is None:
                continue
            for criterion_name, criterion in self.criterions_last.items():
                loss += criterion(pred_lvl, gt) * self.lambdas_cls[criterion_name]
        return loss


class PixLoss(nn.Module):
    """
    Pixel loss for each refined map output.
    """

    def __init__(self, model):
        super(PixLoss, self).__init__()
        self.config = Config()
        if model == 'ISNet-GT':
            self.lambdas_pix_last = self.config.lambdas_gt_encoder
        elif model == 'ISNet_1':
            self.lambdas_pix_last = self.config.lambdas_isnet_1
        elif model == 'ISNet_2':
            self.lambdas_pix_last = self.config.lambdas_isnet_2

        self.criterions_last = {}
        if 'bce' in self.lambdas_pix_last and self.lambdas_pix_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
        if 'iou' in self.lambdas_pix_last and self.lambdas_pix_last['iou']:
            self.criterions_last['iou'] = IoULoss()
        if 'iou_patch' in self.lambdas_pix_last and self.lambdas_pix_last['iou_patch']:
            self.criterions_last['iou_patch'] = PatchIoULoss()
        if 'mse' in self.lambdas_pix_last and self.lambdas_pix_last['mse']:
            self.criterions_last['mse'] = nn.MSELoss()
        if 'reg' in self.lambdas_pix_last and self.lambdas_pix_last['reg']:
            self.criterions_last['reg'] = ThrReg_loss()
        if 'cnt' in self.lambdas_pix_last and self.lambdas_pix_last['cnt']:
            self.criterions_last['cnt'] = ContourLoss()

    def execute(self, scaled_preds, gt):
        loss = 0.
        for _, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.interpolate(pred_lvl, size=gt.shape[2:], mode='bilinear', align_corners=True)
            pred_lvl = pred_lvl.sigmoid()
            for criterion_name, criterion in self.criterions_last.items():
                _loss = criterion(pred_lvl, gt) * self.lambdas_pix_last[criterion_name]
                loss += _loss
                # print(criterion_name, _loss.item())
        return loss


# class SSIMLoss(nn.Module):
# SSIM部分代码待施工
