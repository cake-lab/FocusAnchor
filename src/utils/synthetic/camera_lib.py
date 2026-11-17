"""Thin-lens camera and Gaussian-PSF defocus rendering in plain PyTorch.

Port of the old gauss_psf CUDA kernel (forward pass only), taken from
~/workspace/DualFocus/dataloader/NYU.py. The `power` exponent is FOSSA's
power_exp_psf kernel (princeton-vl/FOSSA, BSD-3), which is the same kernel with
(r/c)^2 raised to power/2. Runs on CPU or GPU.
"""
import torch
import torch.nn.functional as F


def gaussPsf(image, sigma, kernel_size=7, power=2.0):
    """Spatially varying generalized-Gaussian blur. image [M,C,H,W], sigma [M,H,W] -> [M,C,H,W].

    For each pixel p with sigma[p] > 1:
        out[p] = sum_q g(sigma[q], q-p) * image[q] / sum_q g(sigma[q], q-p)
    over the kernel_size x kernel_size neighbours q inside the image with sigma[q] > 1,
    where g(c, dx, dy) = exp(-2 ((dx^2 + dy^2) / c^2)^(power/2)) / c^2.
    power = 2 is a Gaussian. Large powers come close to a disc.
    Pixels with sigma[p] <= 1 are copied through.
    """
    _, _, h, w = image.shape
    mid = kernel_size // 2
    # Zero padding gives out-of-image taps sigma 0, so they fail the "> 1" test,
    # same as the kernel's bounds check.
    sigma_pad = F.pad(sigma, (mid, mid, mid, mid))
    image_pad = F.pad(image, (mid, mid, mid, mid))
    num = torch.zeros_like(image)
    den = torch.zeros_like(sigma)
    for i in range(kernel_size):
        dy = mid - i
        for j in range(kernel_size):
            dx = mid - j
            s = sigma_pad[:, i:i + h, j:j + w]
            valid = s > 1
            s2 = torch.where(valid, s * s, torch.ones_like(s))
            r2 = dx * dx + dy * dy
            # The Gaussian keeps the old expression, so the test stacks stay bit-identical.
            arg = -2.0 * r2 / s2 if power == 2 else -2.0 * (r2 / s2) ** (power / 2)
            g = torch.where(valid, torch.exp(arg) / s2, torch.zeros_like(s))
            den = den + g
            num = num + g[:, None] * image_pad[:, :, i:i + h, j:j + w]
    blur = num / den.clamp_min(1e-30)[:, None]
    return torch.where((sigma > 1)[:, None], blur, image)


class ThinLenCamera:
    def __init__(
        self,
        fnumber=0.5,
        focal_length=2.9 * 1e-3,
        sensor_size=3.1 * 1e-3,
        img_size=560,
        pixel_size=None,
    ):
        self.focal_length = focal_length
        self.D = self.focal_length / fnumber
        self.pixel_size = pixel_size
        if not self.pixel_size:
            self.pixel_size = sensor_size / img_size

    def getCoC(self, dpt, focus_dist):
        # dpt and focus_dist broadcast against each other.
        sensor_dist = focus_dist * self.focal_length / (focus_dist - self.focal_length)
        CoC = (
            self.D
            * sensor_dist
            * torch.abs(1 / self.focal_length - 1 / sensor_dist - 1 / (dpt + 1e-8))
        )
        sigma = CoC / 2 / self.pixel_size
        return sigma.type(torch.float32)


def renderDefocus(rgb, depth, camera, focus_distances, kernel_size=7, power=2.0):
    """rgb [3,H,W], depth [1,H,W], focus_distances [N] -> focal stack [N,3,H,W]."""
    focus = torch.as_tensor(focus_distances, dtype=torch.float32, device=depth.device)
    sigma = camera.getCoC(depth.float(), focus.view(-1, 1, 1))
    image = rgb.float().expand(len(focus), -1, -1, -1)
    return gaussPsf(image, sigma, kernel_size, power)
