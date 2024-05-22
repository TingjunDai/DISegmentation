import logging
import os
import numpy as np
import random
import cv2
from PIL import Image
from jittor.transform import ToPILImage


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('DIS')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_tensor_img(tenor_im, path):
    im = tenor_im.clone()
    tensor2pil = ToPILImage()
    im = tensor2pil(im)
    im = im.convert('L')
    im.save(path)


def check_state_dict(state_dict, unwanted_prefix='_orig_mod.'):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def split_map(datapath):  # 用于生成UDUN的trunk和struct
    """
    From https://https://github.com/weijun88/LDF/blob/master/utils.py
    """
    print(datapath)
    for name in os.listdir(datapath + '/gt'):
        mask = cv2.imread(datapath + '/gt/' + name, 0)
        body = cv2.blur(mask, ksize=(5, 5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body ** 0.5
        tmp = body[np.where(body > 0)]

        if len(tmp) != 0:
            body[np.where(body > 0)] = np.floor(
                tmp / np.max(tmp) * 255)

        if not os.path.exists(datapath + '/trunk-origin/'):
            os.makedirs(datapath + '/trunk-origin/')
        cv2.imwrite(datapath + '/trunk-origin/' + name, body)

        if not os.path.exists(datapath + '/struct-origin/'):
            os.makedirs(datapath + '/struct-origin/')
        cv2.imwrite(datapath + '/struct-origin/' + name, mask - body)


def split_head_and_base(net):
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    return base, head


if __name__ == '__main__':
    split_map('../../DIS5K/DIS5K/DIS-TR')
