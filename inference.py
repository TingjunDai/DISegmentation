import os
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import jittor as jt
from jittor import init
from jittor import nn
from models.isnet.isnet import ISNet
from models.udun.udun import UDUN
from models.birefnet.birefnet import BiRefNet
from models.mvanet.mvanet import MVANet
from config import Config
from utils import save_tensor_img, check_state_dict
from dataset import get_data_loader

config = Config()

jt.flags.use_cuda = 1


def inference(model, data_loader_test, pred_root, method, testset):
    model_training = model.is_training()
    if model_training:
        model.eval()
    for batch in tqdm(data_loader_test, total=len(data_loader_test)):
        inputs = batch[0]
        label_paths = batch[-1]
        with jt.no_grad():
            if config.model == 'ISNet':
                scaled_preds = model(inputs)[0][0].sigmoid()
            elif config.model == 'UDUN':
                scaled_preds = model(inputs)[2].sigmoid()
            elif config.model == 'BiRefNet':
                scaled_preds = model(inputs)[-1].sigmoid()
            elif config.model == 'MVANet':
                scaled_preds = model(inputs).sigmoid()
        os.makedirs(os.path.join(pred_root, method, testset), exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            res = nn.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),
                size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                mode='bilinear',
                align_corners=True
            )
            save_tensor_img(res, os.path.join(os.path.join(pred_root, method, testset), label_paths[idx_sample].replace('\\', '/').split('/')[-1]))   # test set dir + file name
    if model_training:
        model.train()
    return None


def main(args):
    print('Inference with model {}'.format(config.model))

    if config.model == 'ISNet':
        model = ISNet()
    elif config.model == 'UDUN':
        model = UDUN(bb_pretrained=False)
    elif config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False)
    elif config.model == 'MVANet':
        model = MVANet(bb_pretrained=False)
    weights_lst = sorted(
        glob(os.path.join(args.ckpt_folder, '*.pkl')) if args.ckpt_folder else [args.ckpt],
        key=lambda x: int(x.split('ep')[-1].split('.pkl')[0]),
        reverse=True
    )
    for testset in args.testsets.split('+'):
        print('>>>> Testset: {}...'.format(testset))
        data_loader_test = get_data_loader(testset, batch_size=config.batch_size_valid, is_train=False)
        for weights in weights_lst:
            if int(weights.strip('.pkl').split('ep')[-1]) % 1 != 0:
                continue
            print('\tInferencing {}...'.format(weights))
            # model.load_state_dict(torch.load(weights, map_location='cpu'))
            state_dict = jt.load(weights)
            # state_dict = check_state_dict(state_dict)
            # model.load_state_dict(state_dict)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            inference(
                model, data_loader_test=data_loader_test, pred_root=args.pred_root,
                method='{}'.format(weights.split('/')[-1]),
                testset=testset
            )

if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', type=str, help='model folder')
    parser.add_argument('--ckpt_folder', default=os.path.join('ckpt', config.model), type=str, help='model folder')
    parser.add_argument('--pred_root', default='e_mvanet', type=str, help='Output folder')
    parser.add_argument('--testsets',
                        default=config.testsets.replace(',', '+'),
                        type=str,
                        help="Test all sets: DIS5K -> 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'")

    args = parser.parse_args()

    main(args)
