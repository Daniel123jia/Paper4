# -*- coding: utf-8 -*-
"""DN4-V2 SAR few-shot ablation research prototype.

Replace DEFAULT_VARIANT when distributing separate network.py files.
The 2019 DN4 I2C core is preserved with the *uploaded implementation's*
additional pairwise-difference Top-k weights. These additions are proposed
research adaptations, not source code from MANEGM or a verified SOTA model.

query: [B,C,H,W]; supports: length N-way list of [shot,C,H,W];
returns scores [B,N-way], or diagnostics with return_details=True.
"""
from __future__ import annotations
import functools
import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

DEFAULT_VARIANT = 'srm_v2'


def weights_init_normal(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.normal_(module.weight, 0., .02)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) and module.affine:
        init.normal_(module.weight, 1., .02)
        init.zeros_(module.bias)


def weights_init_xavier(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.xavier_normal_(module.weight, gain=.02)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) and module.affine:
        init.normal_(module.weight, 1., .02)
        init.zeros_(module.bias)


def weights_init_kaiming(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) and module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def weights_init_orthogonal(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.orthogonal_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) and module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def init_weights(net, init_type='normal'):
    funcs = {'normal': weights_init_normal, 'xavier': weights_init_xavier,
             'kaiming': weights_init_kaiming, 'orthogonal': weights_init_orthogonal}
    if init_type not in funcs:
        raise NotImplementedError('Unrecognized init_type: ' + str(init_type))
    net.apply(funcs[init_type])


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    if norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    if norm_type == 'none':
        return lambda _: nn.Identity()
    raise NotImplementedError('Unrecognized norm type: ' + str(norm_type))


class ImgtoClass_Metric(nn.Module):
    """Preserves the user's Top-k neighbor *difference weighting*.

    Different from plain DN4 where Top-k similarities are simply added.
    Local output: [batch, query_H * query_W, n_way].
    """
    def __init__(self, neighbor_k=3, topk_weighting='original'):
        super().__init__()
        self.neighbor_k = int(neighbor_k)
        self.topk_weighting = topk_weighting
        if self.neighbor_k < 1:
            raise ValueError('neighbor_k must be >= 1')
        if topk_weighting not in ('original', 'mean'):
            raise ValueError('topk_weighting must be original or mean')

    @staticmethod
    def calculate_weights_matrix(values):
        k = values.size(-1)
        if k == 1:
            return torch.ones_like(values)
        delta = (values.unsqueeze(-1) - values.unsqueeze(-2)).abs()
        z = delta.sum(-1)
        total = z.sum(-1, keepdim=True)
        # The uploaded code had an all-zero weight case when k values tied.
        # Use uniform weights in that rare case rather than a zero score.
        uniform = torch.full_like(z, 1. / k)
        return torch.where(total > 1e-8, z / total.clamp_min(1e-8), uniform)

    def local_similarity(self, query, supports, return_neighbors=False):
        if query.ndim != 4:
            raise ValueError('Query features must be [B,C,H,W]')
        if not isinstance(supports, (list, tuple)) or len(supports) < 1:
            raise ValueError('Supports must be a nonempty class-ordered list')
        B, C, H, W = query.shape
        q = F.normalize(query.flatten(2).transpose(1, 2), dim=-1, eps=1e-8)
        scores = []
        spreads = []
        for support in supports:
            if support.ndim != 2 or support.size(0) != C or support.size(1) == 0:
                raise ValueError('Each support feature pool must be [C,shot*H*W]')
            s = F.normalize(support, dim=0, eps=1e-8)
            cos = q @ s
            topk = cos.topk(min(self.neighbor_k, cos.size(-1)), dim=-1).values
            if self.topk_weighting == 'original':
                coef = self.calculate_weights_matrix(topk)
                sim = (coef * topk).sum(-1)
            else:
                sim = topk.mean(-1)
            scores.append(sim)
            if return_neighbors:
                # k-NN corroboration indicator; low variance alone does NOT
                # demonstrate that this is an actual target descriptor.
                spreads.append(topk.std(dim=-1, unbiased=False))
        out = torch.stack(scores, dim=-1)
        if return_neighbors:
            return out, torch.stack(spreads, dim=-1)
        return out

    def cal_cosinesimilarity(self, input1, input2, local_weights=None,
                             return_local=False):
        local = self.local_similarity(input1, input2)
        if local_weights is None:
            score = local.sum(dim=1)
        else:
            if local_weights.shape != local.shape[:2]:
                raise ValueError('local weights must be [B, H*W]')
            w = local_weights
            M = local.size(1)
            score = (local * w.unsqueeze(-1)).sum(1) * (
                M / w.sum(dim=1, keepdim=True).clamp_min(1e-8))
        return (score, local) if return_local else score

    def forward(self, x1, x2, local_weights=None, return_local=False):
        return self.cal_cosinesimilarity(x1, x2, local_weights, return_local)




def _episode_score_scale(local):
    """Typical per-descriptor cross-class spread, per query [B,1].

    This is not a fitted calibration temperature; it makes entropy less
    dependent on arbitrary raw cosine scales across training regimes.
    If classes are identical, near-zero spread yields uniform posteriors.
    """
    return local.std(dim=-1, unbiased=False).mean(dim=1, keepdim=True).clamp_min(1e-5)


def _softmax_episode_mean(local, tau):
    raw = local.mean(dim=1)
    scale = _episode_score_scale(local).detach()
    centered = raw - raw.mean(dim=-1, keepdim=True)
    return (centered / (scale * tau)).softmax(dim=-1)

class StructuralReliabilityV2(nn.Module):
    """SRM-V2: input-resolution point responses + edges + Conv features.

    Unlike V1, localized strong echoes are retained as *candidate* cues,
    instead of only downsampling an edge/coherence map before extraction.
    The output is a trainable gate, not a ground-truth scattering map.
    """
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(channels + 5, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
        self.register_buffer('sobel_x', sx.reshape(1, 1, 3, 3))
        self.register_buffer('sobel_y', sx.transpose(0, 1).contiguous().reshape(1, 1, 3, 3))

    @staticmethod
    def avg(x, k):
        return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)

    def structural_cues(self, image, target_size):
        # Work on input-grid before downsampling: preserves peaks that could
        # disappear if one first interpolated a high-resolution SAR frame.
        gray = image.mean(dim=1, keepdim=True)
        gray = torch.sign(gray) * torch.log1p(gray.abs())
        gray = (gray - gray.mean((2, 3), keepdim=True)) / (
            gray.std((2, 3), keepdim=True, unbiased=False) + .1)
        local3 = self.avg(gray, 3)
        local7 = self.avg(gray, 7)
        std3 = (self.avg(gray.square(), 3) - local3.square()).clamp_min(1e-8).sqrt()
        point_small = ((gray - local3).abs() / (std3 + .25)).clamp(0, 8)
        point_large = ((gray - local7).abs() / (self.avg((gray - local7).abs(), 7) + .25)).clamp(0, 8)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = (gx.square() + gy.square() + 1e-8).sqrt()
        jxx = self.avg(gx.square(), 3)
        jyy = self.avg(gy.square(), 3)
        jxy = self.avg(gx * gy, 3)
        coherence = ((jxx - jyy).square() + 4 * jxy.square() + 1e-8).sqrt() / (
            jxx + jyy + .1)
        # Both max-pooled and average-pooled high-pass cues: isolated point
        # echoes and distributed edge patterns can both reach the predictor.
        out = torch.cat((
            F.adaptive_max_pool2d(point_small, target_size),
            F.adaptive_avg_pool2d(point_small, target_size),
            F.adaptive_max_pool2d(point_large, target_size),
            F.adaptive_avg_pool2d(grad, target_size),
            F.adaptive_avg_pool2d(coherence.clamp(0, 1), target_size),
        ), dim=1)
        return out

    def forward(self, feature, image):
        cues = self.structural_cues(image, feature.shape[-2:])
        return self.pred(torch.cat((feature, cues.to(feature.dtype)), dim=1)).flatten(1)


class TaskDiscriminabilityV2(nn.Module):
    """TDM-V2: local class contrast + neighbor corroboration + agreement.

    All features are calculated in the *current episode*.
    A high local class margin can still be wrong; neither neighbor consensus
    nor global agreement guarantees correctness.
    """
    def __init__(self, hidden=16, tau_local=1.0, tau_global=.6,
                 neighbor_tau=1.0):
        super().__init__()
        if min(tau_local, tau_global, neighbor_tau) <= 0:
            raise ValueError('TDM temperatures must be positive')
        self.tau_local = float(tau_local)
        self.tau_global = float(tau_global)
        self.neighbor_tau = float(neighbor_tau)
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, local, neighbor_spread):
        if local.size(-1) < 2:
            raise ValueError('TDM needs >= 2 episode classes')
        n = local.size(-1)
        scale = _episode_score_scale(local).detach()  # [B,1]
        centered = local - local.mean(dim=-1, keepdim=True)
        p = (centered / (self.tau_local * scale.unsqueeze(-1))).softmax(-1)
        h = -(p * p.clamp_min(1e-8).log()).sum(-1) / math.log(n)
        top = local.topk(2, dim=-1)
        high, second = top.values[..., 0], top.values[..., 1]
        margin = (high - second).clamp_min(0) / (scale + 1e-5)
        high_class = top.indices[..., 0].unsqueeze(-1)
        spread = neighbor_spread.gather(-1, high_class).squeeze(-1)
        corroboration = torch.exp(-spread / ((scale + 1e-5) * self.neighbor_tau))
        # Cross-descriptor consensus of class posteriors; this favors match
        # patterns compatible with aggregate evidence but is not a label.
        global_prob = _softmax_episode_mean(local, self.tau_global).detach()
        gp = global_prob.unsqueeze(1).expand_as(p)
        meanp = (p + gp) * .5
        js = .5 * (p * (p.clamp_min(1e-8) / meanp.clamp_min(1e-8)).log()).sum(-1)
        js = js + .5 * (gp * (gp.clamp_min(1e-8) / meanp.clamp_min(1e-8)).log()).sum(-1)
        agreement = (1. - js / math.log(2)).clamp(0, 1)
        stats = torch.stack((1. - h, margin, (high + 1.) * .5,
                             corroboration, agreement), dim=-1)
        return self.mlp(stats).squeeze(-1)


class LocalMaskHead(nn.Module):
    """Only in UAG-only control, never inserted into the fusion variants."""
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, feature):
        return self.pred(feature).flatten(1)


class UncertaintyGateV2(nn.Module):
    """Sample entropy and split-descriptor prediction disagreement.

    Compared with V1, a *low-entropy but internally inconsistent* query
    receives more conservative masking. This is NOT a calibrated uncertainty
    guarantee; temperatures and gate statistics need validation.
    """
    def __init__(self, tau_global=.6, alpha_min=.10, alpha_max=.65,
                 gamma=1., agreement_power=2.,
                 use_entropy=True, use_agreement=True, fixed_alpha=.4):
        super().__init__()
        if tau_global <= 0 or gamma <= 0 or agreement_power < 0:
            raise ValueError('Invalid UAG temperature/exponents')
        if not (0 <= alpha_min <= alpha_max < 1):
            raise ValueError('Require 0 <= alpha_min <= alpha_max < 1')
        if not (0 <= fixed_alpha < 1):
            raise ValueError('Require 0 <= fixed_alpha < 1')
        self.tau_global = float(tau_global)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.gamma = float(gamma)
        self.agreement_power = float(agreement_power)
        self.use_entropy = bool(use_entropy)
        self.use_agreement = bool(use_agreement)
        self.fixed_alpha = float(fixed_alpha)

    def forward(self, local, hw):
        B, M, N = local.shape
        if N < 2:
            raise ValueError('UAG needs >= 2 classes')
        if hw[0] * hw[1] != M:
            raise ValueError('Incorrect H*W for UAG local scores')
        scale = _episode_score_scale(local).detach()
        prob = _softmax_episode_mean(local, self.tau_global)
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(-1) / math.log(N)
        # Spatial checkerboard split: no extra forward pass or augmentation.
        H, W = hw
        yy = torch.arange(H, device=local.device).view(H, 1)
        xx = torch.arange(W, device=local.device).view(1, W)
        sel = ((yy + xx) % 2 == 0).reshape(M)
        if not sel.any() or (~sel).sum() == 0:
            split_agreement = torch.ones_like(entropy)
        else:
            # IMPORTANT: both halves use the SAME whole-query scale,
            # so this agreement measure is not distorted by different
            # local temperature normalizations in each partition.
            raw1 = local[:, sel, :].mean(1)
            raw2 = local[:, ~sel, :].mean(1)
            p1 = ((raw1 - raw1.mean(-1, keepdim=True)) /
                  (scale * self.tau_global)).softmax(-1)
            p2 = ((raw2 - raw2.mean(-1, keepdim=True)) /
                  (scale * self.tau_global)).softmax(-1)
            m = (p1 + p2) * .5
            js = .5 * (p1 * (p1.clamp_min(1e-8) / m.clamp_min(1e-8)).log()).sum(-1)
            js += .5 * (p2 * (p2.clamp_min(1e-8) / m.clamp_min(1e-8)).log()).sum(-1)
            split_agreement = (1. - js / math.log(2)).clamp(0, 1)
        if not self.use_entropy:
            alpha = entropy.new_full((B,), self.fixed_alpha)
        else:
            certainty = (1. - entropy.detach()).clamp(0, 1).pow(self.gamma)
            if self.use_agreement:
                certainty = certainty * split_agreement.detach().pow(self.agreement_power)
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * certainty
        return alpha, entropy, split_agreement


class _BaseDN4(nn.Module):
    def _init_metric(self, channels, variant, neighbor_k,
                     topk_weighting='original', fixed_alpha=.4,
                     fusion_lambda=.4, tau_local=1.0, tau_global=.6,
                     neighbor_tau=1.0, alpha_min=.1, alpha_max=.65,
                     gamma=1., agreement_power=2.,
                     use_entropy=True, use_agreement=True):
        allowed = ('baseline', 'srm_v2', 'tdm_v2', 'uag_v2',
                   'plain_mask', 'uag_entropy_only',
                   'srm_tdm_v2', 'srm_tdm_uag_v2')
        if variant not in allowed:
            raise ValueError('Unknown variant: ' + str(variant))
        if not (0 <= fixed_alpha < 1) or not (0 <= fusion_lambda <= 1):
            raise ValueError('fixed_alpha in [0,1), fusion_lambda in [0,1]')
        self.variant = variant
        self.fixed_alpha = float(fixed_alpha)
        self.fusion_lambda = float(fusion_lambda)
        self.imgtoclass = ImgtoClass_Metric(neighbor_k, topk_weighting)
        if variant in ('srm_v2', 'srm_tdm_v2', 'srm_tdm_uag_v2'):
            self.srm = StructuralReliabilityV2(channels)
        if variant in ('tdm_v2', 'srm_tdm_v2', 'srm_tdm_uag_v2'):
            self.tdm = TaskDiscriminabilityV2(
                tau_local=tau_local, tau_global=tau_global,
                neighbor_tau=neighbor_tau)
        if variant in ('uag_v2', 'plain_mask', 'uag_entropy_only'):
            self.local_mask = LocalMaskHead(channels)
        if variant in ('uag_v2', 'plain_mask', 'uag_entropy_only',
                       'srm_tdm_uag_v2'):
            self.uag = UncertaintyGateV2(
                tau_global=tau_global, alpha_min=alpha_min,
                alpha_max=alpha_max, gamma=gamma,
                agreement_power=agreement_power,
                use_entropy=(variant not in ('plain_mask',) and use_entropy),
                use_agreement=(variant not in ('plain_mask', 'uag_entropy_only')
                               and use_agreement), fixed_alpha=fixed_alpha)

    def _features(self, x):
        if hasattr(self, 'features'):
            return self.features(x)
        return self.feat_extractor(x)

    def forward(self, input1, input2, return_details=False):
        if not isinstance(input2, (list, tuple)) or len(input2) == 0:
            raise ValueError('Support must be a nonempty list of per-class tensors')
        q = self._features(input1)
        supports = []
        for s in input2:
            if s.ndim != 4:
                raise ValueError('Each class support must be [shot,C,H,W]')
            feature = self._features(s)
            supports.append(feature.permute(1, 0, 2, 3).contiguous().reshape(feature.size(1), -1))
        use_tdm = hasattr(self, 'tdm')
        if use_tdm:
            local, spread = self.imgtoclass.local_similarity(q, supports, return_neighbors=True)
        else:
            local = self.imgtoclass.local_similarity(q, supports)
            spread = None
        initial_logits = local.sum(1)
        extras = {}
        reliability = None
        a = None
        d = None
        if hasattr(self, 'srm'):
            a = self.srm(q, input1)
            extras['structure_reliability'] = a
        if hasattr(self, 'tdm'):
            d = self.tdm(local, spread)
            extras['task_discriminability'] = d
        if a is not None and d is not None:
            reliability = a * (self.fusion_lambda + (1. - self.fusion_lambda) * d)
            extras['combined_reliability'] = reliability
        elif a is not None:
            reliability = a
        elif d is not None:
            reliability = d
        elif hasattr(self, 'local_mask'):
            reliability = self.local_mask(q)
            extras['base_local_mask'] = reliability
        if reliability is None:
            weights = None
            logits = initial_logits
        else:
            if hasattr(self, 'uag'):
                alpha, h, agree = self.uag(local, q.shape[-2:])
                extras.update({'entropy': h, 'alpha': alpha,
                               'split_agreement': agree})
            else:
                alpha = reliability.new_full((reliability.size(0),), self.fixed_alpha)
                extras['alpha'] = alpha
            weights = 1. - alpha.unsqueeze(-1) * (1. - reliability)
            M = local.size(1)
            logits = (local * weights.unsqueeze(-1)).sum(1)
            logits = logits * (M / weights.sum(1, keepdim=True).clamp_min(1e-8))
        if return_details:
            return dict(logits=logits, initial_logits=initial_logits,
                        local_scores=local, descriptor_weights=weights, **extras)
        return logits

    def srm_consistency_loss(self, original_query, weak_query):
        """Optional SRM mask consistency for pixel-ALIGNED weak transforms.

        This is not automatically included in the user's existing training loop.
        The two images need matching spatial positions and same class label.
        """
        if not hasattr(self, 'srm'):
            raise RuntimeError('This variant does not include SRM')
        if original_query.shape != weak_query.shape:
            raise ValueError('Aligned views must have the same shape')
        q1 = self._features(original_query)
        q2 = self._features(weak_query)
        a1 = self.srm(q1, original_query)
        a2 = self.srm(q2, weak_query)
        return F.mse_loss(a1, a2)


class FourLayer_64F(_BaseDN4):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5,
                 neighbor_k=3, variant=DEFAULT_VARIANT, in_channels=3, **kwargs):
        super().__init__()
        use_bias = (norm_layer is nn.InstanceNorm2d or
                    (isinstance(norm_layer, functools.partial) and
                     norm_layer.func is nn.InstanceNorm2d))
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=use_bias),
            norm_layer(64), nn.LeakyReLU(.2, inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1, bias=use_bias),
            norm_layer(64), nn.LeakyReLU(.2, inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1, bias=use_bias),
            norm_layer(64), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=use_bias),
            norm_layer(64), nn.LeakyReLU(.2, inplace=True),
        )
        self._init_metric(64, variant, neighbor_k, **kwargs)


class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super().__init__()
        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(.2))
        self.conv_block.add_module('ConvL1', nn.Conv2d(nFin, nFout, 3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(.2))
        self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, 3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(.2))
        self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, 3, padding=1, bias=False))
        self.skip_layer = nn.Conv2d(nFin, nFout, 1, 1)

    def forward(self, x):
        return self.skip_layer(x) + self.conv_block(x)


class ResNetLike(_BaseDN4):
    def __init__(self, opt=None, neighbor_k=3, variant=DEFAULT_VARIANT,
                 in_channels=3, **kwargs):
        super().__init__()
        chans = [64, 96, 128, 256]
        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0', nn.Conv2d(in_channels, chans[0], 3, padding=1))
        for i, c in enumerate(chans):
            self.feat_extractor.add_module('ResBlock' + str(i),
                ResBlock(chans[i - 1] if i > 0 else chans[0], c))
            if i < 2:
                self.feat_extractor.add_module('MaxPool' + str(i), nn.MaxPool2d(2, 2))
        self.feat_extractor.add_module('ReluF1', nn.LeakyReLU(.2, inplace=True))
        self._init_metric(256, variant, neighbor_k, **kwargs)


def define_DN4Net(pretrained=False, model_root=None, which_model='Conv64',
                  norm='batch', init_type='normal', use_gpu=True,
                  variant=DEFAULT_VARIANT, **kwargs):
    if which_model in ('Conv64', 'Conv64F'):
        model = FourLayer_64F(norm_layer=get_norm_layer(norm), variant=variant, **kwargs)
    elif which_model == 'ResNet256F':
        model = ResNetLike(variant=variant, **kwargs)
    else:
        raise NotImplementedError('Unrecognized backbone: ' + str(which_model))
    init_weights(model, init_type=init_type)
    if pretrained:
        if isinstance(model_root, (str, bytes, os.PathLike)):
            try:
                state = torch.load(model_root, map_location='cpu', weights_only=True)
            except TypeError:
                state = torch.load(model_root, map_location='cpu')
        elif isinstance(model_root, dict):
            state = model_root
        else:
            raise ValueError('pretrained=True expects path or state_dict')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise ValueError('Expected model_root to contain a state_dict')
        # Previous V1 SRM and TDM heads have different input dimensions.
        # PyTorch strict=False still raises on shape mismatch, so deliberately
        # skip incompatible head tensors while reporting every skipped key.
        current = model.state_dict()
        candidates = {
            (k[7:] if k.startswith('module.') else k): v
            for k, v in state.items() if torch.is_tensor(v)
        }
        compatible = {k: v for k, v in candidates.items()
                      if k in current and tuple(v.shape) == tuple(current[k].shape)}
        skipped = [k for k, v in candidates.items()
                   if k not in current or tuple(v.shape) != tuple(current[k].shape)]
        if not compatible:
            raise ValueError('No compatible model tensor was found in the checkpoint')
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print('Compatible checkpoint tensors:', len(compatible),
              '| skipped (missing key/shape):', skipped,
              '| newly initialized tensors:', missing)
        if skipped:
            warnings.warn('Some checkpoint tensors were not loaded. '
                          'The changed V2 module heads require retraining.', RuntimeWarning)
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; set use_gpu=False')
        model.cuda()
    return model


def print_network(model):
    print(model)
    print('Total number of parameters: %d' % sum(x.numel() for x in model.parameters()))
