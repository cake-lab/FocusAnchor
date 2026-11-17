from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.synthetic import camera_lib


# ImageNet statistics, matching what the precomputed-stack datasets already apply.
FOCAL_STACK_MEAN = (0.485, 0.456, 0.406)
FOCAL_STACK_STD = (0.229, 0.224, 0.225)

SYNTHETIC_CAMERAS = {
    "nyu": camera_lib.ThinLenCamera(8, focal_length=50e-3, pixel_size=1.2e-5),
}

# FOSSA's training render (Zuo et al., "Zero-Shot Depth from Defocus", 2026), ported from
# princeton-vl/FOSSA (BSD-3): util/gen_focal_stack.py, util/util.py, and the random
# scaling in dataset/base.py.
FOSSA_FNUMBERS = (1.0, 1.4, 2.0, 2.8, 4.0)
FOSSA_SENSOR_WIDTH = 36e-3  # a full-frame sensor across the long image edge
FOSSA_KERNEL_SIZE = 15
# Mean of fx and fy in pixels, for the frame the loader returns. NYU uses the Kinect RGB
# intrinsics. The centre crop to 384x512 does not change them.
FOSSA_FOCAL_PIXELS = {"nyu": (518.8579 + 519.4696) / 2}


class DepthBatch(NamedTuple):
    """Normalized supervised batch consumed by the Lightning module."""

    rgb_aif: torch.Tensor
    focal_stack: torch.Tensor | None
    depth_gt: torch.Tensor
    focus_dist: torch.Tensor
    mask: torch.Tensor
    synthetic_camera: str | Sequence[str] | None = None


def coerceDepthBatch(batch: Mapping[str, Any] | Sequence[Any]) -> DepthBatch:
    """Accept current dataset outputs and expose one named batch shape. The loaders
    return dicts, except DDFF12, which returns a 5-tuple."""
    if isinstance(batch, Mapping):
        required_keys = {"rgb_aif", "depth", "focus_dist", "mask"}
        missing_keys = required_keys.difference(batch)
        if missing_keys:
            raise ValueError(
                "Expected a mapping batch with keys "
                f"{sorted(required_keys)}; missing {sorted(missing_keys)}."
            )

        focal_stack = batch.get("focal_stack")
        synthetic_camera = batch.get("synthetic_camera")
        if focal_stack is None and synthetic_camera is None:
            raise ValueError(
                "Mapping batches must include either 'focal_stack' or "
                "'synthetic_camera'."
            )

        return DepthBatch(
            rgb_aif=batch["rgb_aif"],
            focal_stack=focal_stack,
            depth_gt=batch["depth"],
            focus_dist=batch["focus_dist"],
            mask=batch["mask"],
            synthetic_camera=synthetic_camera,
        )

    if len(batch) != 5:
        raise ValueError(
            "Expected a supervised tuple batch with 5 items: "
            "rgb_aif, focal_stack, depth_gt, focus_dist, mask."
        )

    return DepthBatch(
        rgb_aif=batch[0],
        focal_stack=batch[1],
        depth_gt=batch[2],
        focus_dist=batch[3],
        mask=batch[4],
    )


def prepareBatch(batch: Mapping[str, Any] | Sequence[Any], fossa: bool = False, noise: float = 0.0,
                 nStack: int | None = None) -> DepthBatch:
    """One batch shape for the Lightning module, with a synthetic focal stack
    rendered for the datasets (NYU) that do not ship one. `fossa`
    renders it with FOSSA's random camera and focus planes instead. `noise` is the
    largest sensor-noise sigma added to the rendered slices (normalizeStack). `nStack`
    is the number of slices the FOSSA render makes, the loader's count when None."""
    depth_batch = coerceDepthBatch(batch)
    if fossa:
        if depth_batch.focal_stack is not None:
            raise ValueError("FOSSA synthesis needs a dataset without a real focal stack (NYU).")
        return renderFossaBatch(depth_batch, noise, nStack)
    if depth_batch.focal_stack is not None:
        return depth_batch
    return depth_batch._replace(focal_stack=renderSyntheticFocalStack(depth_batch, noise))


def renderSyntheticFocalStack(batch: DepthBatch, noise: float = 0.0) -> torch.Tensor:
    """Render a defocus stack from the all-in-focus RGB and the GT depth. This runs
    on the training device rather than in a DataLoader worker, so the render uses
    the GPU when there is one."""
    rgb_aif = batch.rgb_aif
    camera_names = batch.synthetic_camera
    if isinstance(camera_names, str):
        camera_names = [camera_names] * len(rgb_aif)
    if len(camera_names) != len(rgb_aif):
        raise ValueError(
            f"Expected {len(rgb_aif)} synthetic camera names, got {len(camera_names)}."
        )

    with torch.no_grad():
        focal_stacks = []
        for sample_idx, camera_name in enumerate(camera_names):
            camera = SYNTHETIC_CAMERAS.get(str(camera_name))
            if camera is None:
                raise ValueError(f"Unknown synthetic camera profile: {camera_name}")

            focal_stacks.append(
                camera_lib.renderDefocus(
                    rgb_aif[sample_idx],
                    batch.depth_gt[sample_idx],
                    camera,
                    batch.focus_dist[sample_idx],
                )
            )

        focal_stack = torch.stack(focal_stacks, dim=0)

    return normalizeStack(focal_stack, noise)


def renderFossaBatch(batch: DepthBatch, noise: float = 0.0, nStack: int | None = None) -> DepthBatch:
    """FOSSA's training render. Per image: a 1-1.5x zoom (it scales the focal length),
    an f-number, a PSF shape and new focus planes, then a 15x15 render. The zoom changes
    the image, depth and mask, and the planes replace the loader's GT-range ones, so
    this returns a new batch.

    FOSSA draws the f-number, PSF and plane mode once per batch. Here each image draws
    its own, so fnet's batch norm does not see one camera per batch."""
    cameraNames = batch.synthetic_camera
    if isinstance(cameraNames, str):
        cameraNames = [cameraNames] * len(batch.rgb_aif)
    n = nStack or batch.focus_dist.shape[1]
    h, w = batch.depth_gt.shape[-2:]
    device = batch.depth_gt.device
    rgbs, depths, masks, planes, stacks = [], [], [], [], []
    with torch.no_grad():
        for i, name in enumerate(cameraNames):
            if name not in FOSSA_FOCAL_PIXELS:
                raise ValueError(f"No FOSSA intrinsics for synthetic camera {name}.")
            # Zoom in, then centre-crop back to h x w (FOSSA's random scaling). If the crop
            # keeps no valid depth (NYU train image 11157 at large zooms), the image is not
            # zoomed, so it still has GT. FOSSA's synthetic depth has no such holes.
            # nearest-exact samples pixel centres like PIL's NEAREST in FOSSA. Plain
            # "nearest" would shift depth about half a pixel against the bilinear RGB.
            for zoom in (1 + 0.5 * torch.rand(()).item(), 1.0):
                zh, zw = int(h * zoom), int(w * zoom)
                top, left = round((zh - h) / 2), round((zw - w) / 2)
                crop = (..., slice(top, top + h), slice(left, left + w))
                mask = F.interpolate(batch.mask[i:i + 1].float(), (zh, zw), mode="nearest-exact")[crop][0] > 0.5
                if mask.any():
                    break
            rgb = F.interpolate(batch.rgb_aif[i:i + 1], (zh, zw), mode="bilinear", align_corners=False)[crop][0]
            depth = F.interpolate(batch.depth_gt[i:i + 1], (zh, zw), mode="nearest-exact")[crop][0]

            pixel = FOSSA_SENSOR_WIDTH / max(h, w)
            focal = pixel * FOSSA_FOCAL_PIXELS[name] * max(zh / h, zw / w)
            fnumber = FOSSA_FNUMBERS[torch.randint(len(FOSSA_FNUMBERS), ()).item()]
            camera = camera_lib.ThinLenCamera(fnumber, focal_length=focal, pixel_size=pixel)

            # Focus planes, near to far. FOSSA lists them far to near.
            if torch.rand(()).item() < 0.2:
                # From the image: 5th to 95th percentile, even in disparity. Over every
                # pixel, as FOSSA's get_depth_dependent_fd_list, including the ones the
                # loss ignores. Only NYU's zero holes are left out.
                disparity = 1 / depth[depth > 0].double()
                lo, hi = torch.quantile(disparity, torch.tensor([0.05, 0.95], dtype=torch.float64, device=device))
                dist = 1 / torch.linspace(hi.item(), lo.item(), n, dtype=torch.float64)
            else:
                # Not from the image: near in [0.6, 1] m, far 7-15x farther, power-law
                # spacing in disparity. kappa = 1 is even in disparity, kappa -> 0 even in
                # log depth. kappa stays above 0, where 1 / kappa would be inf.
                near = 0.6 + 0.4 * torch.rand((), dtype=torch.float64)
                far = near * (7 + 8 * torch.rand((), dtype=torch.float64))
                kappa = torch.rand((), dtype=torch.float64).clamp(min=1e-6)
                t = torch.linspace(0, 1, n, dtype=torch.float64)
                dist = 1 / ((1 - t) * near ** -kappa + t * far ** -kappa) ** (1 / kappa)
            dist = dist.float().to(device)

            # PSF exponent 2^U(1,5): 2 is a Gaussian, 32 is close to a disc.
            power = 2 ** (1 + 4 * torch.rand(()).item())
            stacks.append(camera_lib.renderDefocus(rgb, depth, camera, dist, FOSSA_KERNEL_SIZE, power))
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            planes.append(dist)

    return batch._replace(rgb_aif=torch.stack(rgbs), depth_gt=torch.stack(depths), mask=torch.stack(masks),
                          focus_dist=torch.stack(planes), focal_stack=normalizeStack(torch.stack(stacks), noise))


def normalizeStack(focalStack: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
    """ImageNet-normalize a rendered stack [B,N,3,H,W] in [0, 1]. `noise` > 0 first adds
    Gaussian sensor noise to every slice on its own, sigma per image in U(0, noise). The
    render copies the AiF's own noise sharp into the in-focus slice and blurs it away in
    the others, so on a blank wall the noise is the focus cue. Real cameras put fresh
    noise on every slice (ZEDD: 0.002 in these units), so the cue is false there."""
    if noise > 0:
        sigma = noise * torch.rand(focalStack.shape[0], 1, 1, 1, 1, device=focalStack.device)
        focalStack = (focalStack + sigma * torch.randn_like(focalStack)).clamp(0, 1)
    mean = torch.tensor(FOCAL_STACK_MEAN, device=focalStack.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(FOCAL_STACK_STD, device=focalStack.device).view(1, 1, 3, 1, 1)
    return (focalStack - mean) / std


def allInFocus(stack):
    """Per pixel, the slice of an RGB uint8 stack [N,H,W,3] with the strongest local
    Laplacian. For the datasets with no all-in-focus image (ZEDD, Infinigen, rendered stacks)."""
    sharpness = np.stack([
        cv2.blur(np.abs(cv2.Laplacian(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), cv2.CV_32F)), (9, 9))
        for s in stack])
    return np.take_along_axis(stack, sharpness.argmax(0)[None, ..., None], axis=0)[0]
