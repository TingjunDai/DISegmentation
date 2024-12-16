import jittor as jt
from jittor import init
from jittor import nn
from jittor.einops import rearrange
from config import Config
import math
from models.backbones.build_backbone import build_backbone

jt.flags.use_cuda = 1

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.relu
    if activation == "gelu":
        return nn.gelu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm(out_dim), nn.PReLU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm(out_dim), nn.GELU())


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return nn.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return nn.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x


class PositionEmbeddingSine():

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if ((scale is not None) and (normalize is False)):
            raise ValueError('normalize should be True if scale is passed')
        if (scale is None):
            scale = (2 * math.pi)
        self.scale = scale
        self.dim_t = jt.arange(0, self.num_pos_feats, dtype=jt.float32)

    def __call__(self, b, h, w):
        mask = jt.zeros([b, h, w], dtype=jt.bool)
        assert (mask is not None)
        not_mask = mask.logical_not()
        y_embed = not_mask.cumsum(dim=1)
        x_embed = not_mask.cumsum(dim=2)
        if self.normalize:
            eps = 1e-06
            y_embed = (((y_embed - 0.5) / (y_embed[:, (- 1):, :] + eps)) * self.scale)
            x_embed = (((x_embed - 0.5) / (x_embed[:, :, (- 1):] + eps)) * self.scale)
        dim_t = (self.temperature ** ((2 * (self.dim_t // 2)) / self.num_pos_feats))
        pos_x = (x_embed[:, :, :, None] / dim_t)
        pos_y = (y_embed[:, :, :, None] / dim_t)
        pos_x = jt.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(start_dim=3)
        pos_y = jt.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(start_dim=3)
        return jt.contrib.concat((pos_y, pos_x), dim=3).permute((0, 3, 1, 2))


class MCLM(nn.Module):

    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(MCLM, self).__init__()
        self.attention = nn.ModuleList([
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])
        self.linear1 = nn.Linear(d_model, (d_model * 2))
        self.linear2 = nn.Linear((d_model * 2), d_model)
        self.linear3 = nn.Linear(d_model, (d_model * 2))
        self.linear4 = nn.Linear((d_model * 2), d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('relu')
        self.pool_ratios = pool_ratios
        self.p_poses = []
        self.g_pos = None
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=(d_model // 2), normalize=True)

    def execute(self, l, g):
        (b, c, h, w) = l.shape
        concated_locs = rearrange(l, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
        pools = []
        if not self.training:
            self.p_poses = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round((h / pool_ratio)), round((w / pool_ratio)))
            pool2d = nn.AdaptiveAvgPool2d(tgt_hw)
            pool = pool2d(concated_locs)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            if self.training:
                if (self.g_pos is None):
                    pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3])
                    pos_emb = rearrange(pos_emb, 'b c h w -> (h w) b c')
                    self.p_poses.append(pos_emb)
            else:
                pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3])
                pos_emb = rearrange(pos_emb, 'b c h w -> (h w) b c')
                self.p_poses.append(pos_emb)
        pools = jt.contrib.concat(pools, dim=0)
        if self.training:
            if (self.g_pos is None):
                self.p_poses = jt.contrib.concat(self.p_poses, dim=0)
                pos_emb = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
                self.g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')
        else:
            self.p_poses = jt.contrib.concat(self.p_poses, dim=0)
            pos_emb = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
            self.g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')
        g_hw_b_c = (g_hw_b_c + self.dropout1(self.attention[0]((g_hw_b_c + self.g_pos), (pools + self.p_poses), pools)[0]))
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = (g_hw_b_c + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(g_hw_b_c)).clone()))))
        g_hw_b_c = self.norm2(g_hw_b_c)
        l_hw_b_c = rearrange(l, 'b c h w -> (h w) b c')
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c, '(ng h) (nw w) b c -> (h w) (ng nw b) c', ng=2, nw=2)
        outputs_re = []
        for (i, (_l, _g)) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[(i + 1)](_l, _g, _g)[0])
        outputs_re = jt.contrib.concat(outputs_re, dim=1)
        l_hw_b_c = (l_hw_b_c + self.dropout1(outputs_re))
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = (l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone()))))
        l_hw_b_c = self.norm2(l_hw_b_c)
        l = jt.contrib.concat((l_hw_b_c, g_hw_b_c), dim=1)
        return rearrange(l, '(h w) b c -> b c h w', h=h, w=w)


class MCRM(nn.Module):

    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(MCRM, self).__init__()
        self.attention = nn.ModuleList([
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
            jt.attention.MultiheadAttention(d_model, num_heads, dropout=0.1),
        ])
        self.linear3 = nn.Linear(d_model, (d_model * 2))
        self.linear4 = nn.Linear((d_model * 2), d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('relu')
        self.sal_conv = nn.Conv(d_model, 1, 1)
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=(d_model // 2), normalize=True)

    def execute(self, x):
        (b, c, h, w) = x.shape
        (loc, glb) = x.split([4, 1], dim=0)
        patched_glb = rearrange(glb, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = nn.interpolate(token_attention_map, size=patches2image(loc).shape[(- 2):], mode='nearest')
        loc = (loc * rearrange(token_attention_map, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2))
        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round((h / pool_ratio)), round((w / pool_ratio)))
            pool2d = nn.AdaptiveAvgPool2d(tgt_hw)
            pool = pool2d(patched_glb)
            pools.append(rearrange(pool, 'nl c h w -> nl c (h w)'))
        pools = rearrange(jt.contrib.concat(pools, dim=2), 'nl c nphw -> nl nphw 1 c')
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')
        outputs = []
        for (i, q) in enumerate(loc_.unbind(dim=0)):
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])
        outputs = jt.contrib.concat(outputs, dim=1)
        src = (loc.view(4, c, (- 1)).permute((2, 0, 1)) + self.dropout1(outputs))
        src = self.norm1(src)
        src = (src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone()))))
        src = self.norm2(src)
        src = src.permute((1, 2, 0)).reshape((4, c, h, w))
        glb = (glb + nn.interpolate(patches2image(src), size=glb.shape[(- 2):], mode='nearest'))
        return (jt.contrib.concat((src, glb), dim=0), token_attention_map)

class MVANet(nn.Module):

    def __init__(self, bb_pretrained=True):
        super().__init__()
        self.config = Config()
        self.backbone = build_backbone(self.config.mva_bb, pretrained=bb_pretrained)
        channel = self.config.mva_lateral_channels_in_collection
        emb_dim = 128
        self.sideout5 = nn.Sequential(nn.Conv(emb_dim, 1, 3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv(emb_dim, 1, 3, padding=1))
        self.sideout3 = nn.Sequential(nn.Conv(emb_dim, 1, 3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv(emb_dim, 1, 3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv(emb_dim, 1, 3, padding=1))
        self.output5 = make_cbr(channel[0], emb_dim)
        self.output4 = make_cbr(channel[1], emb_dim)
        self.output3 = make_cbr(channel[2], emb_dim)
        self.output2 = make_cbr(channel[3], emb_dim)
        self.output1 = make_cbr(channel[3], emb_dim)
        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8])
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8])
        self.insmask_head = nn.Sequential(nn.Conv(emb_dim, 384, 3, padding=1), nn.BatchNorm(384), nn.PReLU(), nn.Conv(384, 384, 3, padding=1), nn.BatchNorm(384), nn.PReLU(), nn.Conv(384, emb_dim, 3, padding=1))
        self.shallow = nn.Sequential(nn.Conv(3, emb_dim, 3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv(emb_dim, 1, 3, padding=1))
        for m in self.modules():
            if (isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout)):
                m.inplace = True

    def execute(self, x):
        shallow = self.shallow(x)
        glb = rescale_to(x, scale_factor=0.5, interpolation='bilinear')
        loc = image2patches(x)
        input = jt.contrib.concat((loc, glb), dim=0)
        feature = self.backbone(input)
        e5 = self.output5(feature[4])
        e4 = self.output4(feature[3])
        e3 = self.output3(feature[2])
        e2 = self.output2(feature[1])
        e1 = self.output1(feature[0])
        (loc_e5, glb_e5) = e5.split([4, 1], dim=0)
        e5 = self.multifieldcrossatt(loc_e5, glb_e5)
        (e4, tokenattmap4) = self.dec_blk4((e4 + resize_as(e5, e4)))
        e4 = self.conv4(e4)
        (e3, tokenattmap3) = self.dec_blk3((e3 + resize_as(e4, e3)))
        e3 = self.conv3(e3)
        (e2, tokenattmap2) = self.dec_blk2((e2 + resize_as(e3, e2)))
        e2 = self.conv2(e2)
        (e1, tokenattmap1) = self.dec_blk1((e1 + resize_as(e2, e1)))
        e1 = self.conv1(e1)
        (loc_e1, glb_e1) = e1.split([4, 1], dim=0)
        output1_cat = patches2image(loc_e1)
        output1_cat = (output1_cat + resize_as(glb_e1, output1_cat))
        final_output = self.insmask_head(output1_cat)
        final_output = (final_output + resize_as(shallow, final_output))
        final_output = self.upsample1(rescale_to(final_output))
        final_output = rescale_to((final_output + resize_as(shallow, final_output)))
        final_output = self.upsample2(final_output)
        final_output = self.output(final_output)
        sideout5 = self.sideout5(e5)
        sideout4 = self.sideout4(e4)
        sideout3 = self.sideout3(e3)
        sideout2 = self.sideout2(e2)
        sideout1 = self.sideout1(e1)
        glb5 = self.sideout5(glb_e5)
        glb4 = sideout4[(- 1), :, :, :].unsqueeze(0)
        glb3 = sideout3[(- 1), :, :, :].unsqueeze(0)
        glb2 = sideout2[(- 1), :, :, :].unsqueeze(0)
        glb1 = sideout1[(- 1), :, :, :].unsqueeze(0)
        sideout1 = patches2image(sideout1[:(- 1)])
        sideout2 = patches2image(sideout2[:(- 1)])
        sideout3 = patches2image(sideout3[:(- 1)])
        sideout4 = patches2image(sideout4[:(- 1)])
        sideout5 = patches2image(sideout5[:(- 1)])
        if self.training:
            return (sideout5, sideout4, sideout3, sideout2, sideout1, final_output, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1)
        else:
            return final_output