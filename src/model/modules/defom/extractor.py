from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.depth_anything_v2.dinov2_layers.drop_path import DropPath
from model.modules.depth_anything_v2.dpt import DepthAnythingV2
from model.modules.depth_anything_v2.util.blocks import _make_scratch

CHECKPOINT_DIR = Path(__file__).resolve().parents[4] / "checkpoints"


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        norms = {'group': lambda: nn.GroupNorm(num_groups=planes // 8, num_channels=planes),
                 'batch': lambda: nn.BatchNorm2d(planes),
                 'instance': lambda: nn.InstanceNorm2d(planes),
                 'none': lambda: nn.Sequential()}
        self.norm1 = norms[norm_fn]()

    def forward(self, x):
        return self.relu(self.norm1(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, d_dim, output_dim=128, norm_fn='batch', downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # depth feat convolution. None for the focal-slice encoder: DAv2 runs on the AiF only.
        self.convd = ConvBlock(d_dim, 128, self.norm_fn) if d_dim else None

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dfeats=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.convd is not None:
            x = x + self.convd(dfeats)

        return self.conv2(x)


class MultiBasicEncoder(nn.Module):
    def __init__(self, d_dim, output_dim=[128, 128, 128], norm_fn='batch', downsample=3, drop_path_rate=0.2):
        super(MultiBasicEncoder, self).__init__()
        self.d_dim = d_dim
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.drop_path = DropPath(drop_path_rate)

        self.conv08 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        self.conv16 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        self.conv32 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, d_feats, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feat = x + self.drop_path(self.conv08(d_feats[0]))
        outputs08 = [f(feat) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08,)

        y = self.layer4(x)
        feat = y + self.drop_path(self.conv16(d_feats[1]))
        outputs16 = [f(feat) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16)

        z = self.layer5(y)
        feat = z + self.drop_path(self.conv32(d_feats[2]))
        outputs32 = [f(feat) for f in self.outputs32]

        return (outputs08, outputs16, outputs32)


class DPTFeat(nn.Module):
    """DAv2 features at 1/4, 1/8, 1/16 of the input for the context encoder (cnet).

    DEFOM also fuses a 4th level into one map for its stereo matching encoder. The focus
    volume uses CNN features only, so that fusion path is gone. Layer names match
    DPTHead, so the weights can be copied from it.
    """

    def __init__(self, in_channels, features=256, out_channels=[256, 512, 1024, 1024]):
        super(DPTFeat, self).__init__()
        out_channels = out_channels[:3]
        self.projects = nn.ModuleList([nn.Conv2d(in_channels, c, kernel_size=1) for c in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
        ])
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)

    def forward(self, out_features, patch_h, patch_w, out_h, out_w):
        layers = [self.scratch.layer1_rn, self.scratch.layer2_rn, self.scratch.layer3_rn]
        out = []
        for i, (x, _) in enumerate(out_features[:3]):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = layers[i](self.resize_layers[i](x))
            out.append(F.interpolate(x, (out_h // 2**i, out_w // 2**i), mode="bilinear", align_corners=True))
        return out


class DefomEncoder(nn.Module):
    """Frozen Depth Anything V2 on the AiF image: relative inverse depth plus DPT features
    for cnet. `dinov2_encoder` is 'vits', 'vitb' or 'vitl'."""

    def __init__(self, dinov2_encoder):
        super(DefomEncoder, self).__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        config = model_configs[dinov2_encoder]
        self.depth_anything = DepthAnythingV2(**config)
        dim = self.depth_anything.pretrained.embed_dim

        # DEFOM skips a missing file silently and trains on a random DAv2. Fail instead.
        path = CHECKPOINT_DIR / f'depth_anything_v2_{dinov2_encoder}.pth'
        if not path.exists():
            raise FileNotFoundError(f'{path} is missing. See README.md for the download.')
        self.depth_anything.load_state_dict(torch.load(path, map_location='cpu'))
        # depth_feat has the same layers as depth_head minus the output convs, so warm-start
        # it from there. DEFOM leaves it at random init. It hangs off depth_anything so the
        # checkpoint keys stay depth_anything.depth_feat.*.
        self.depth_anything.depth_feat = DPTFeat(dim, config['features'], config['out_channels'])
        missing = self.depth_anything.depth_feat.load_state_dict(self.depth_anything.depth_head.state_dict(),
                                                                 strict=False).missing_keys
        assert not missing, missing

        self.depth_anything.pretrained.requires_grad_(False)
        self.depth_anything.depth_head.requires_grad_(False)
        self.out_dim = config['features']

    def forward(self, x, danv2_io_sizes):
        ih, iw, oh, ow = danv2_io_sizes
        x = F.interpolate(x, (ih, iw), mode="bilinear", align_corners=True)
        m, ph, pw = self.depth_anything, ih // 14, iw // 14
        features = m.pretrained.get_intermediate_layers(x, m.intermediate_layer_idx[m.encoder],
                                                        return_class_token=True)
        head = m.depth_head
        scratch, rn = head.scratch, []
        for (f, _), project, resize, layer in zip(features, head.projects, head.resize_layers,
                                                  [scratch.layer1_rn, scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn]):
            f = project(f.permute(0, 2, 1).reshape(f.shape[0], -1, ph, pw))
            rn.append(layer(resize(f)))
        # DPTHead.forward as DEFOM changed it: the last fusion block does not upsample x2,
        # and the output lands on (oh, ow) instead of the input size.
        path = scratch.refinenet4(rn[3], size=rn[2].shape[2:])
        path = scratch.refinenet3(path, rn[2], size=rn[1].shape[2:])
        path = scratch.refinenet2(path, rn[1], size=rn[0].shape[2:])
        path = scratch.refinenet1(path, rn[0], size=rn[0].shape[2:])
        path = F.interpolate(scratch.output_conv1(path), (oh, ow), mode="bilinear", align_corners=True)
        idepth = scratch.output_conv2(path)
        return m.depth_feat(features, ph, pw, oh, ow), idepth
