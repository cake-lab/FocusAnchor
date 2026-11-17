"""ZEDD (Zuo et al., "Zero-Shot Depth from Defocus", ECCV 2026) with FOSSA's protocol.
Ported from princeton-vl/FOSSA dataset/zedd.py (BSD-3).

50 scenes from a Sony a1 II with a 50 mm lens: 9 focus distances x 6 apertures, JPEGs at
4104x2736, lidar GT in metres at 1824x1216 with +inf where there is none. FOSSA's
protocol: F/2.8, runs 1, 3, 5, 7, 9 (0.82 to 8.10 m), LANCZOS RGB and NEAREST depth to
480x720, centre crop to 480x640, GT kept where finite and in [0.01, 1000] m.

The test split hides its GT behind FOSSA's server, which scores the full frame at
1216x1824. `crop=False` keeps the whole frame, and `height=1216` keeps the native
1216x1824, as FOSSA's test loader does. Test scenes come with a zero depth and an empty
mask.

ZEDD has no all-in-focus image. `aif="fused"` takes the sharpest F/2.8 slice per pixel,
so the model sees the same images as FOSSA. `aif="f16"` is the F/16 shot focused at 3.08 m,
which FOSSA's loader reads but its model never sees.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataloader.batch import FOCAL_STACK_MEAN, FOCAL_STACK_STD, allInFocus

FNUMBER = 2.8
RUNS = (1, 3, 5, 7, 9)
HEIGHT, WIDTH = 480, 640
MIN_DEPTH, MAX_DEPTH = 0.01, 1000.0


class ZEDDLoader(Dataset):
    def __init__(self, root: str, aif: str = "fused", split: str = "val", crop: bool = True, height: int = HEIGHT):
        if aif not in ("fused", "f16"):
            raise ValueError("ZEDD aif must be 'fused' or 'f16'.")
        self.root = Path(root)
        self.aif = aif
        self.crop = crop
        self.height = height
        self.scenes = sorted((self.root / split).iterdir())
        # metadata.txt: run_id, motor hex code, focus distance in metres, tab separated.
        rows = [line.split("\t") for line in (self.root / "metadata.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        self.hexCode = {run: code for run, code, _ in rows}
        distance = {run: float(d) for run, _, d in rows}
        self.focus_dist = torch.tensor([distance[f"run_{k}"] for k in RUNS])

    def __len__(self):
        return len(self.scenes)

    def resize(self, image, interpolation):
        """FOSSA's resize: height to `self.height`, then the centre 640 columns if cropping."""
        width = int(image.width * self.height / image.height)
        left = (width - WIDTH) // 2 if self.crop else 0
        return np.asarray(image.resize((width, self.height), interpolation))[:, left:left + (WIDTH if self.crop else width)]

    def readImage(self, path):
        return self.resize(Image.open(path).convert("RGB"), Image.LANCZOS)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        stack = np.stack([self.readImage(scene / "focus_stack" / f"img_run_{k}_motor_{self.hexCode[f'run_{k}']}_aperture_F{FNUMBER:.1f}.jpg")
                          for k in RUNS])
        aif = allInFocus(stack) if self.aif == "fused" else self.readImage(scene / "focus_stack" / "img_run_5_motor_2371_aperture_F16.0.jpg")

        if (scene / "gt" / "depth.npy").exists():
            depth = self.resize(Image.fromarray(np.load(scene / "gt" / "depth.npy").astype(np.float32)), Image.NEAREST)
        else:
            depth = np.zeros(stack.shape[1:3], np.float32)
        mask = np.isfinite(depth) & (depth >= MIN_DEPTH) & (depth <= MAX_DEPTH)
        depth = np.where(mask, depth, 0).astype(np.float32)

        mean = torch.tensor(FOCAL_STACK_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(FOCAL_STACK_STD).view(1, 3, 1, 1)
        return {
            "rgb_aif": torch.from_numpy(aif.copy()).permute(2, 0, 1).float() / 255,
            "focal_stack": (torch.from_numpy(stack.copy()).permute(0, 3, 1, 2).float() / 255 - mean) / std,
            "depth": torch.from_numpy(depth)[None],
            "focus_dist": self.focus_dist.clone(),
            "mask": torch.from_numpy(mask)[None],
        }
