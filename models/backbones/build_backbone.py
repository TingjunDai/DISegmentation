import jittor as jt
from jittor import nn
from models.backbones.swin_v1 import swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l
from models.backbones.resnet50 import resnet50
from config import Config

config = Config()


def build_backbone(bb_name, pretrained=True, params_settings=''):
    bb = eval('{}({})'.format(bb_name, params_settings))
    if pretrained:
        bb = load_weights(bb, bb_name)
    return bb


def load_weights(model, model_name):
    save_model = jt.load(config.backbone_weights[model_name])
    model_dict = model.state_dict()
    state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in save_model.items() if
                  k in model_dict.keys()}
    # to ignore the weights with mismatched size when I modify the backbone itself.
    if not state_dict:
        save_model_keys = list(save_model.keys())
        sub_item = save_model_keys[0] if len(save_model_keys) == 1 else None
        state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in
                      save_model[sub_item].items() if k in model_dict.keys()}
        if not state_dict or not sub_item:
            print('Weights are not successully loaded. Check the state dict of weights file.')
            return None
        else:
            print('Found correct weights in the "{}" item of loaded state_dict.'.format(sub_item))
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model
