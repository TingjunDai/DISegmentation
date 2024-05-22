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
    im = im.data
    tensor2pil = ToPILImage()
    print(im)
    im = tensor2pil(im)
    im = im.convert('L')
    im.save(path)


def check_state_dict(state_dict, unwanted_prefix='_orig_mod.'):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict
