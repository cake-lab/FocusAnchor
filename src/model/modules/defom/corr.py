import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinearSampler(img, coords):
    """grid_sample with pixel coordinates. From DEFOM core/utils/utils.py. Zeros outside."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


class FocusVolume(nn.Module):
    """Focus volume from the slices only, as in DFV. No AiF: the synthetic stacks copy
    in-focus pixels from the AiF, so matching slices against it is too easy.

    Each slice feature goes in with its difference to the next slice (DFV's differential
    volume). 3D convs over (slice, y, x) give one focus score per slice and pixel. The
    kernel is 3 along the stack, so any N works.
    """

    def __init__(self, dim=32):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(2 * dim, dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(dim, 1, 3, padding=1))

    def forward(self, fStack):
        """Slice features [B,N,C,h,w] -> focus scores [B,N,h,w]."""
        f = fStack.transpose(1, 2)  # [B, C, N, h, w]
        # DFV: slice n minus slice n+1, and the last slice minus 0.
        diff = torch.cat([f[:, :, :-1] - f[:, :, 1:], f[:, :, -1:]], dim=2)
        return self.convs(torch.cat([f, diff], dim=1)).squeeze(1)


class FocusBlock:
    """Focus volume over the stack, read at a continuous inverse-depth hypothesis.

    This replaces DEFOM's CorrBlock1D. In stereo, the lookup reads how well the left
    pixel matches each right pixel along the line. Here it reads the focus score of
    each slice (`FocusVolume`). The slice focused at a pixel's depth scores highest.

    Positions along the stack are slice coordinates u (0 = nearest plane). `invFocus`
    maps them to inverse depth, piecewise linear between planes and extended linearly
    outside the stack, so any N and any plane spacing work.
    """

    def __init__(self, volume, invFocus, radius=4, levels=2, step=0.5,
                 scaleList=(0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0), scaleRadius=2):
        b, n, h, w = volume.shape
        self.volume = volume.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, n)
        self.shape = (b, h, w)
        self.n = n
        self.invFocus = invFocus  # [B, N], near to far, so descending
        self.ascending = invFocus.flip(1).contiguous()
        self.scaleList = scaleList

        # DEFOM avg-pools the volume for its coarser levels. With 5 slices that leaves
        # almost nothing, so the coarser level uses dilated taps on the same volume.
        taps = torch.arange(-radius, radius + 1, device=volume.device, dtype=torch.float32) * step
        self.taps = [taps * 2**i for i in range(levels)]
        self.scaleTaps = torch.arange(-scaleRadius, scaleRadius + 1, device=volume.device,
                                      dtype=torch.float32) * step

    def toIndex(self, x):
        """Inverse depth [B,1,h,w] -> slice coordinate u."""
        b, _, h, w = x.shape
        flat = x.reshape(b, -1).contiguous()
        k = torch.searchsorted(self.ascending, flat).clamp(1, self.n - 1)
        lo, hi = self.ascending.gather(1, k - 1), self.ascending.gather(1, k)
        u = (k - 1) + (flat - lo) / (hi - lo).clamp(min=1e-6)
        return (self.n - 1 - u).view(b, 1, h, w)

    def toInverse(self, u):
        """Slice coordinate [B,1,h,w] -> inverse depth."""
        k = u.detach().floor().clamp(0, self.n - 2).long()
        lo = self.invFocus.gather(1, k.flatten(1)).view_as(u)
        hi = self.invFocus.gather(1, (k + 1).flatten(1)).view_as(u)
        return lo + (u - k) * (hi - lo)

    def sample(self, u, taps):
        b, h, w = self.shape
        x0 = u.reshape(-1, 1, 1, 1) + taps.view(1, 1, -1, 1)
        out = bilinearSampler(self.volume, torch.cat([x0, torch.zeros_like(x0)], dim=-1))
        return out.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

    def __call__(self, x, scaling=False):
        if scaling:
            # Scale lookup: read the volume around s * x for every candidate scale s.
            return torch.cat([self.sample(self.toIndex(s * x), self.scaleTaps) for s in self.scaleList], dim=1)
        u = self.toIndex(x)
        return torch.cat([self.sample(u, taps) for taps in self.taps], dim=1)
