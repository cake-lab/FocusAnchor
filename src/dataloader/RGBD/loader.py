"""iBims-1, HAMMER and ARKitScenes with FOSSA's Table 3 protocol (Zuo et al., "Zero-Shot Depth from
Defocus", ECCV 2026). Ported from princeton-vl/FOSSA dataset/uniformat.py, dataset/hammer.py
and util/gen_focal_stack.py (BSD-3).

All are real RGB-D datasets with no focal stack, so FOSSA renders one from the GT at test
time: F/1.4, a Gaussian PSF (exponent 2) with a 15x15 kernel, a 36 mm sensor across the long
edge with the dataset's fx and fy, and 5 planes even in disparity from the 5th to the 95th
percentile of the image's disparity. The percentiles are over every pixel of the depth the
render uses. Native resolution, no resize.

- iBims-1: 100 images at 480x640 from FOSSA's Uniformat release (venkatsubra/Uniformat on
  Hugging Face), with the holes filled by OMNI-DC. The render uses the filled depth, the
  metrics only the valid pixels.
- HAMMER: 775 frames at 832x1856 from MoGe-2's evaluation set
  (Ruicheng/monocular-geometry-evaluation). Pixels with no depth are 1000 m for the render, as
  in FOSSA, and are left out of the metrics.
- ARKitScenes: not in FOSSA's table. The 5,594 validation images at 480x640, read by
  ARKitscenesDataset, with the per-frame intrinsics scaled from 256x192. The laser-scan holes
  (up to 40% of an image) are filled with the nearest depth for the render only.

The AiF is the sharpest slice per pixel, as on ZEDD and Infinigen, so the model sees the same
images as FOSSA.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from dataloader.ARKitScenes.labeled import ARKitscenesDataset
from dataloader.batch import FOSSA_KERNEL_SIZE, FOSSA_SENSOR_WIDTH, allInFocus, normalizeStack
from utils.synthetic import camera_lib

FNUMBER = 1.4
N_PLANES = 5


class NumpyUnpickler(pickle.Unpickler):
    """Each iBims file is a pickled dict of arrays. The files are downloads and a plain
    pickle can run code, so only numpy's array types may load."""

    def find_class(self, module, name):
        if (module, name) in {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed in an iBims file")


class RGBDLoader(Dataset):
    def __init__(self, root: str, dataset: str = "ibims"):
        if dataset not in ("ibims", "hammer", "arkit"):
            raise ValueError("RGBD dataset must be 'ibims', 'hammer' or 'arkit'.")
        self.dataset = dataset
        root = self.root = Path(root)
        if dataset == "ibims":
            self.samples = sorted(root.glob("*.npy"))
        elif dataset == "arkit":
            self.arkit = ARKitscenesDataset(str(root), img_size=(480, 640), stage="Validation")
            self.samples = list(range(len(self.arkit)))
        else:
            self.samples = [root / line.strip() for line in (root / ".index.txt").read_text().splitlines() if line.strip()]
        if not self.samples:
            raise FileNotFoundError(f"No {dataset} samples under {root}")

    def __len__(self):
        return len(self.samples)

    def read(self, idx):
        """RGB uint8 [H,W,3], the depth the render uses, the GT mask and K."""
        read = {"ibims": self.readIbims, "hammer": self.readHammer, "arkit": self.readArkit}[self.dataset]
        return read(self.samples[idx])

    def readIbims(self, path):
        with open(path, "rb") as f:
            np.lib.format.read_magic(f)  # all 100 files are .npy version 1.0
            np.lib.format.read_array_header_1_0(f)
            data = NumpyUnpickler(f).load().item()
        # FOSSA transposes iBims' K to match the other datasets.
        return data["rgb"], data["depth_filled"], data["valid_mask"].astype(bool), data["K"].T

    def readHammer(self, frame):
        rgb = np.asarray(Image.open(frame / "image.jpg").convert("RGB"))
        h, w = rgb.shape[:2]
        # The intrinsics are normalized by the image size.
        K = np.array(json.loads((frame / "meta.json").read_text())["intrinsics"]) * [[w], [h], [1]]
        # MoGe's depth PNG: 16 bits of log depth between near and far, 0 = none, 65535 = inf.
        png = Image.open(frame / "depth.png")
        raw = np.asarray(png)
        t = (raw.astype(np.float32) - 1) / 65533
        depth = float(png.info["near"]) ** (1 - t) * float(png.info["far"]) ** t * float(png.info.get("unit", 1))
        mask = (raw != 0) & (raw != 65535) & (depth > 0)
        return rgb, np.where(mask, depth, 1000), mask, K

    def readArkit(self, i):
        color, depth = self.arkit[i]
        rgb = color.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
        depth = depth[0].numpy()
        mask = depth > 0
        nearest = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        # The .pincam file is "w h fx fy cx cy" for the 256x192 frame of the same camera.
        path = self.arkit.samples_pth[i].split()[0].replace("lowres_wide", "lowres_wide_intrinsics").replace(".png", ".pincam")
        w, h, fx, fy, cx, cy = map(float, (self.root / "Validation" / path).read_text().split()[:6])
        K = np.array([[fx * 640 / w, 0, cx * 640 / w], [0, fy * 480 / h, cy * 480 / h], [0, 0, 1]])
        return rgb, depth[tuple(nearest)], mask, K

    def __getitem__(self, idx):
        rgb, depth, mask, K = self.read(idx)
        # The render runs on the GPU: on one CPU thread a HAMMER frame takes 20 s. So the
        # datamodule keeps num_workers at 0, since CUDA does not work in DataLoader workers.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rgb = torch.from_numpy(np.array(rgb)).to(device).permute(2, 0, 1).float() / 255
        depth = torch.from_numpy(depth.astype(np.float32)).to(device)[None]
        mask = torch.from_numpy(mask)[None]

        # Planes near to far. FOSSA lists the same planes far to near.
        lo, hi = torch.quantile(1 / depth.flatten(), torch.tensor([0.05, 0.95], device=device))
        focus = 1 / torch.linspace(hi.item(), lo.item(), N_PLANES)
        pixel = FOSSA_SENSOR_WIDTH / max(depth.shape[-2:])
        camera = camera_lib.ThinLenCamera(FNUMBER, focal_length=pixel * float(K[0, 0] + K[1, 1]) / 2, pixel_size=pixel)
        stack = camera_lib.renderDefocus(rgb, depth, camera, focus, FOSSA_KERNEL_SIZE).cpu()
        depth, focus = depth.cpu(), focus.cpu()

        aif = allInFocus((stack.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))
        return {
            "rgb_aif": torch.from_numpy(aif.copy()).permute(2, 0, 1).float() / 255,
            "focal_stack": normalizeStack(stack[None])[0],
            "depth": torch.where(mask, depth, 0),
            "focus_dist": focus,
            "mask": mask,
        }
