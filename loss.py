import jittor as jt
from jittor import nn
from config import Config
from math import exp

import numpy as np

config = Config()

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
        if config.model == 'UDUN':
            return IoU/b
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
    
    
class StructureLoss(nn.Module):

    def __init__(self):
        super(StructureLoss, self).__init__()

    def execute(self, pred, target):
        _target = nn.pad(target, [15, 15, 15, 15])
        # print(target.size())
        weit = (1 + (5 * jt.abs_((nn.avg_pool2d(_target, kernel_size=31, stride=1, padding=0) - target))))
        wbce = nn.binary_cross_entropy_with_logits(pred, target)
        wbce = ((weit * wbce).sum((2, 3)) / weit.sum((2, 3)))
        
        pred  = jt.sigmoid(pred)
        inter = ((pred * target) * weit).sum((2, 3))
        union = ((pred + target) * weit).sum((2, 3))
        wiou = (1 - ((inter + 1) / ((union - inter) + 1)))
        return (wbce + wiou).mean()


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
    def __init__(self, model):
        super(PixLoss, self).__init__()
        self.config = Config()
        if model == 'ISNet':
            self.lambdas_pix_last = self.config.lambdas_isnet
        elif model == 'BiRefNet':
            self.lambdas_pix_last = self.config.lambdas_birefnet
        elif model == 'MVANet':
            self.lambdas_pix_last = self.config.lambdas_mvanet

        self.criterions_last = {}
        if 'bce' in self.lambdas_pix_last and self.lambdas_pix_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
        if 'bce_logits' in self.lambdas_pix_last and self.lambdas_pix_last['bce_logits']:
            self.criterions_last['bce_logits'] = nn.BCEWithLogitsLoss()
        if 'iou' in self.lambdas_pix_last and self.lambdas_pix_last['iou']:
            self.criterions_last['iou'] = IoULoss()
        if 'iou_patch' in self.lambdas_pix_last and self.lambdas_pix_last['iou_patch']:
            self.criterions_last['iou_patch'] = PatchIoULoss()
        if 'ssim' in self.lambdas_pix_last and self.lambdas_pix_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mae' in self.lambdas_pix_last and self.lambdas_pix_last['mae']:
            self.criterions_last['mae'] = nn.L1Loss()
        if 'mse' in self.lambdas_pix_last and self.lambdas_pix_last['mse']:
            self.criterions_last['mse'] = nn.MSELoss()
        if 'reg' in self.lambdas_pix_last and self.lambdas_pix_last['reg']:
            self.criterions_last['reg'] = ThrReg_loss()
        if 'cnt' in self.lambdas_pix_last and self.lambdas_pix_last['cnt']:
            self.criterions_last['cnt'] = ContourLoss()
        if 'structure' in self.lambdas_pix_last and self.lambdas_pix_last['structure']:
            self.criterions_last['structure'] = StructureLoss()

    def execute(self, scaled_preds, gt):
        loss = 0.
        for _, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.interpolate(pred_lvl, size=gt.shape[2:], mode='bilinear', align_corners=True)
            # pred_lvl = pred_lvl.sigmoid()
            for criterion_name, criterion in self.criterions_last.items():
                if criterion_name != 'structure' and criterion_name != 'bce_logits':
                    _loss = criterion(pred_lvl.sigmoid(), gt) * self.lambdas_pix_last[criterion_name]
                    loss += _loss
                else:
                    _loss = criterion(pred_lvl, gt) * self.lambdas_pix_last[criterion_name]
                    loss += _loss
                # print(criterion_name, _loss.item())
        return loss


class SupLoss(nn.Module):
    def __init__(self):
        super(SupLoss, self).__init__()
        self.config = Config()
        self.lambdas_cls = {
            'mse': 1 * 1, 
        }

        self.criterions_last = {
            'mse': nn.MSELoss()
        }

    def execute(self, preds, gt):
        loss = 0.
        for i in range(0, len(preds)):
            for criterion_name, criterion in self.criterions_last.items():
                loss += criterion(preds[i], gt[i]) * self.lambdas_cls[criterion_name]
        return loss

class SSIMLoss(nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def execute(self, img1, img2):
        (_, channel, _, _) = img1.shape
        if ((channel == self.channel) and (self.window.dtype == img1.dtype)):
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return (1 - ((1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)) / 2))


def gaussian(window_size, sigma):
    gauss = jt.array([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.matmul(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = nn.conv(img1, window, padding = window_size//2, groups=channel)
    mu2 = nn.conv(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = nn.conv(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = nn.conv(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = nn.conv(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return jt.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x, y):
    ssim = SSIM(x,y).mean()
    return ssim

