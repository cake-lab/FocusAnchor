import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.defom.corr import FocusBlock, FocusVolume
from model.modules.defom.extractor import BasicEncoder, DefomEncoder, MultiBasicEncoder
from model.modules.defom.update import BasicMultiUpdateBlock, ScaleBasicMultiUpdateBlock

# Floor for the normalized inverse depth: 100x farther than the nearest focus plane,
# or 4x beyond the far plane when that is farther still.
MIN_INVERSE = 0.01
# Prior fusion: half-width of the local DAv2 fit window at 1/4 resolution, and the
# pseudo-count that pulls a window with few confident pixels back to the global fit.
PRIOR_RADIUS = 16
PRIOR_COUNT = 0.05


def getDanv2IoSize(h, w, nds, maxSize=2688, multipleOf=14):
    """DAv2 input size whose DPT output lands on the 1/2^nds feature grid. From DEFOM."""
    oh, ow = h // 2**nds, w // 2**nds
    ih = int(np.ceil(3.5 * oh / multipleOf) * multipleOf)
    iw = int(np.ceil(3.5 * ow / multipleOf) * multipleOf)
    maxSize = int(np.floor(maxSize / multipleOf) * multipleOf)
    if ih > maxSize or iw > maxSize:
        factor = maxSize / max(ih, iw)
        ih = int(np.ceil(factor * ih / multipleOf) * multipleOf)
        iw = int(np.ceil(factor * iw / multipleOf) * multipleOf)
    return ih, iw, oh, ow


class DefomFocus(nn.Module):
    """DEFOM-Stereo with the stereo pair replaced by an AiF image and its focal stack.
    The AiF goes to DAv2 and cnet only. The focus volume sees the slices only, as in DFV.

    The state is inverse depth normalized by the nearest focus plane, so the stack spans
    [invFocus[:, -1], 1] for every dataset. DAv2 sets the start, the scale iterations fix
    the per-pixel scale, and the delta iterations refine in slice units.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        # rgb_aif comes in [0, 1]. The focal stack is already ImageNet-normalized.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.defomencoder = DefomEncoder(args.dinov2_encoder)
        # CNN-only features for the focus volume. DAv2 features are made to ignore blur,
        # which is the cue we need, so they go to the context encoder only. Batch norm as
        # in DFV: instance norm would rescale each slice on its own and blur the contrast
        # between sharp and defocused slices.
        self.fnet = BasicEncoder(None, output_dim=32, norm_fn='batch', downsample=args.n_downsample)
        self.focusVolume = FocusVolume(dim=32)
        self.cnet = MultiBasicEncoder(self.defomencoder.out_dim, output_dim=[args.hidden_dims, args.hidden_dims],
                                      norm_fn=args.context_norm, downsample=args.n_downsample)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(args.hidden_dims[i], args.hidden_dims[i] * 3, 3, padding=1)
                                                for i in range(args.n_gru_layers)])
        self.update_block = BasicMultiUpdateBlock(args, hidden_dims=args.hidden_dims)
        self.scale_update_block = ScaleBasicMultiUpdateBlock(args, hidden_dims=args.hidden_dims)
        if args.prior_fusion != 'none':
            # Confidence that the current state is right, from the GRU state, the focus
            # lookup and the gap to the fitted DAv2. Trained by BCE in FocusAnchorModule.
            corPlanes = args.corr_levels * (2 * args.corr_radius + 1)
            self.confHead = nn.Sequential(nn.Conv2d(args.hidden_dims[0] + corPlanes + 1, 64, 3, padding=1),
                                          nn.ReLU(inplace=True), nn.Conv2d(64, 1, 3, padding=1))

    def upsample(self, x, mask):
        """Convex upsampling [B,1,h,w] -> [B,1,fh,fw]. DEFOM multiplies by f because
        disparity is in pixels. Inverse depth does not change with resolution."""
        b, _, h, w = x.shape
        f = 2**self.args.n_downsample
        mask = torch.softmax(mask.view(b, 1, 9, f, f, h, w), dim=2)
        up = F.unfold(x, [3, 3], padding=1).view(b, 1, 9, 1, 1, h, w)
        up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(b, 1, f * h, f * w)

    def forward(self, rgbAif, focalStack, invFocus, iters=12, scaleIters=4, testMode=False):
        """rgbAif [B,3,H,W] in [0,1], focalStack [B,N,3,H,W], invFocus [B,N] inverse
        focus distances divided by the nearest one (so invFocus[:, 0] = 1).
        Returns the list of predictions (only the last one in testMode), the DAv2
        initialization and the volume's own DFV-style prediction, all normalized inverse
        depth at full resolution, and the confidence logits of the delta iterations
        (empty without prior fusion)."""
        args = self.args
        b, n, _, h, w = focalStack.shape
        aif = (rgbAif - self.mean) / self.std
        mixed = torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.mixed_precision and aif.is_cuda)

        with mixed:
            dFeatures, idepth = self.defomencoder(aif, getDanv2IoSize(h, w, args.n_downsample))
            cnetList = self.cnet(aif, dFeatures)
            net = [torch.tanh(x[0]) for x in cnetList]
            inp = [torch.relu(x[1]) for x in cnetList]
            # Run the GRU context convs once instead of every iteration.
            inp = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp, self.context_zqr_convs)]
            fStack = self.fnet(focalStack.flatten(0, 1))

        # The volume and its lookups stay in fp32, as in DEFOM.
        invFocus = invFocus.float()
        fStack = fStack.float().view(b, n, *fStack.shape[1:])
        volume = self.focusVolume(fStack)
        # Coarser volumes from pooled slice features, same 3D convs, added on. DFV builds
        # its volume at several scales so evidence reaches textureless regions.
        for k in range(1, args.volume_scales):
            pooled = F.avg_pool2d(fStack.flatten(0, 1), 2**k)
            coarse = self.focusVolume(pooled.view(b, n, *pooled.shape[1:]))
            volume = volume + F.interpolate(coarse, volume.shape[-2:], mode='bilinear', align_corners=True)
        focusFn = FocusBlock(volume, invFocus, radius=args.corr_radius, levels=args.corr_levels,
                             step=args.corr_step, scaleList=args.scale_list, scaleRadius=args.scale_corr_radius)
        # DFV's regression: the expected focus plane under a softmax over slices. Its own
        # loss trains the mean of the softmax, not how peaked it is.
        volumePred = (volume.softmax(1) * invFocus.view(b, n, 1, 1)).sum(1, keepdim=True)
        volumePred = F.interpolate(volumePred, (h, w), mode='bilinear', align_corners=True)

        # DEFOM maps DAv2 inverse depth to [0, idepth_scale * width] pixels. Here it is
        # mapped min-max onto the focus range [far plane, nearest plane].
        idepth = idepth.float()
        lo = idepth.amin(dim=(1, 2, 3), keepdim=True)
        hi = idepth.amax(dim=(1, 2, 3), keepdim=True)
        far = invFocus[:, -1].view(b, 1, 1, 1)
        # A nearest plane close to the camera would make the 100x floor cut off the far
        # planes, so the floor drops to 4x beyond the far plane in that case.
        floor = (far / 4).clamp(max=MIN_INVERSE)
        x = far + (1 - far) * (idepth - lo) / (hi - lo + 1e-8)
        init = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        prior = (idepth - lo) / (hi - lo + 1e-8)  # DAv2 shape in [0, 1], refitted every delta iteration
        conf = torch.ones_like(x)
        confidences = []

        # Delta clip: the span the lookup can see, in slices.
        maxDelta = 2**(args.corr_levels - 1) * args.corr_radius * args.corr_step
        gruFlags = dict(iter32=args.n_gru_layers == 3, iter16=args.n_gru_layers >= 2)
        predictions = []
        for itr in range(iters):
            x = x.detach()
            if itr < scaleIters:
                corr = focusFn(x, scaling=True)
                with mixed:
                    net, upMask, scale = self.scale_update_block(net, inp, corr, x, **gruFlags)
                x = scale.float() * x
            else:
                corr = focusFn(x)
                disp = x
                if args.prior_fusion != 'none':
                    # Fit DAv2 to the state per window, weighted by the last confidence.
                    # Weighted moments with a pseudo-count toward the image-wide fit.
                    with torch.no_grad():
                        moments = torch.cat([conf, conf * prior, conf * x, conf * prior**2, conf * prior * x], 1)
                        local = F.avg_pool2d(moments, 2 * PRIOR_RADIUS + 1, stride=1, padding=PRIOR_RADIUS,
                                             count_include_pad=False)
                        whole = moments.mean((2, 3), keepdim=True)
                        m = (local[:, 1:] + PRIOR_COUNT * whole[:, 1:] / whole[:, :1].clamp(min=1e-6)) / (local[:, :1] + PRIOR_COUNT)
                        ea, ex, eaa, eax = m.split(1, dim=1)
                        s = ((eax - ea * ex) / (eaa - ea**2).clamp(min=1e-6)).clamp(min=0)
                        fitted = s * prior + ex - s * ea
                        # Unclamped fits run to ~0 inverse depth and send pixels to the depth cap.
                        fitted = torch.minimum(torch.maximum(fitted, 0.5 * x.amin((2, 3), keepdim=True)),
                                               2 * x.amax((2, 3), keepdim=True))
                        fitted = torch.maximum(fitted, floor)
                    with mixed:
                        confLogit = self.confHead(torch.cat([net[0], corr, x.log() - fitted.log()], 1))
                    conf = torch.sigmoid(confLogit.float())
                    disp = torch.cat([x, fitted, conf.detach()], 1)
                with mixed:
                    net, upMask, delta = self.update_block(net, inp, corr, disp, **gruFlags)
                delta = delta.float().clamp(-maxDelta, maxDelta)
                x = focusFn.toInverse(focusFn.toIndex(x) + delta)
                if args.prior_fusion == 'blend':
                    x = conf * x + (1 - conf) * fitted
                conf = conf.detach()
            x = torch.maximum(x, floor)

            if testMode and itr < iters - 1:
                continue
            predictions.append(self.upsample(x, upMask.float()))
            if args.prior_fusion != 'none' and itr >= scaleIters:
                confidences.append(F.interpolate(confLogit.float(), (h, w), mode='bilinear', align_corners=True))

        return predictions, init, volumePred, confidences
