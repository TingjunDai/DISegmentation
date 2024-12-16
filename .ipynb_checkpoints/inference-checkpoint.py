import os
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import jittor as jt
from jittor import init
from jittor import nn
from models.isnet_gt import ISNetGTEncoder
from models.isnet import ISNetDIS
from models.udun import UDUN
from models.birefnet import BiRefNet
from config import Config
from utils import save_tensor_img, check_state_dict
from dataset import get_train_loader, get_udun_loader, get_gt_loader, get_birefnet_loader

config = Config()

parent_dir = os.path.dirname(os.getcwd())

jt.flags.use_cuda = 1

def inference(model, data_loader_test, pred_root, method, testset, size):
    model_training = model.is_training()
    model.eval()
    for batch in tqdm(data_loader_test, total=int(data_loader_test.batch_len())):
        inputs = batch[0]
        names = batch[1]
        with jt.no_grad():
            if config.model == 'ISNet':
                scaled_preds = model(inputs)[0][0].sigmoid()
            elif config.model == 'UDUN':
                scaled_preds = model(inputs)[2].sigmoid()
            else:
                scaled_preds = model(inputs)[-1].sigmoid()
        os.makedirs(os.path.join(parent_dir, pred_root, method, testset), exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            res = nn.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),  # squeeze操作满足interpolate输入条件
                size=cv2.imread(names[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                mode='bilinear',
                align_corners=True
            )
            res = res.squeeze(0)
            res = res.squeeze(0)
            ma = jt.max(res)
            mi = jt.min(res)
            result = (res - mi) / (ma - mi) * 255
            save_tensor_img(result, os.path.join(parent_dir, pred_root, method, testset,
                                                 names[idx_sample].replace('\\', '/').split('/')[
                                                     -1]))  # test set dir + file name
    if model_training:
        model.train()
    return None


def main(args):
    print('Inference with model {}'.format(config.model))

    if config.model == 'ISNet':
        model = ISNetDIS()
    elif config.model == 'UDUN':
        model = UDUN()
    elif config.model == 'BiRefNet':
        model = BiRefNet()
    else:
        return
    weights = args.weights
    for testset in args.testsets.split('+'):
        print('>>>> Testset: {}...'.format(testset))
        root_path = os.path.join('../../datasets/dis/DIS5K', testset)
        if config.model == 'ISNet':
            data_loader_test = get_train_loader(image_root=os.path.join(root_path, 'im/'),
                                                gt_root=os.path.join(root_path, 'gt/'),
                                                batchsize=4, trainsize=1024, is_train=False)
        elif config.model == 'UDUN':
            data_loader_test = get_udun_loader(image_root=os.path.join(root_path, 'im/'),
                                               gt_root=os.path.join(root_path, 'gt/'),
                                               batchsize=4, trainsize=1024, is_train=False)
        else:
            data_loader_test = get_birefnet_loader(image_root=os.path.join(root_path, 'im/'),
                                                gt_root=os.path.join(root_path, 'gt/'),
                                                batchsize=4, trainsize=1024, is_train=False)
        print('\tInferencing {}...'.format(weights))
        # model.load_state_dict(torch.load(weights, map_location='cpu'))
        if config.model != 'UDUN':
            state_dict = jt.load(weights)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
        inference(
            model, data_loader_test=data_loader_test, pred_root=args.pred_root,
            method='{}'.format(weights.split('/')[-2]),
            testset=testset, size=1024
        )


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', default='birefnet', type=str, help='model folder')
    parser.add_argument('--weights', default='../saved_model/birefnet/BiRefNet_DIS_ep500-swin_v1_tiny.pth', type=str, help='model folder')
    parser.add_argument('--pred_root', default='inference', type=str, help='Output folder')
    parser.add_argument('--testsets',
                        default={
                            'DIS5K': 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4',
                            'COD': 'COD10K+NC4K+CAMO+CHAMELEON',
                            'SOD': 'DAVIS-S+HRSOD-TE+UHRSD-TE+DUTS-TE+DUT-OMRON',
                            'DIS5K-': 'DIS-VD',
                            'COD-': 'COD10K',
                            'SOD-': 'DAVIS-S+HRSOD-TE+UHRSD-TE',
                        }['DIS5K-'],
                        type=str,
                        help="Test all sets: , 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'")

    args = parser.parse_args()
    main(args)
