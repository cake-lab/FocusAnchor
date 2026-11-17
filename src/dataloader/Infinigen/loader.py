"""Infinigen Defocus (Zuo et al., "Zero-Shot Depth from Defocus", ECCV 2026).
Ported from princeton-vl/FOSSA dataset/infinigen_defocus.py (BSD-3).

200 Infinigen Indoors scenes with defocus rendered by Blender Cycles: 9 focus distances x 5
apertures, PNGs and GT in metres at 720x960, fx = fy = 2000 px. FOSSA's protocol: F/1.4, planes
at 0.8, 1.7, 3.0, 4.7, 8.0 m, GT kept where finite and in [0.01, 1000] m. FOSSA runs at 720x960.
Here everything is resized to 480x640, LANCZOS RGB and NEAREST depth, as FOSSA does on ZEDD.

The AiF is the sharpest F/1.4 slice per pixel, as on ZEDD, so the model sees the same images as FOSSA.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataloader.batch import FOCAL_STACK_MEAN, FOCAL_STACK_STD, allInFocus

FNUMBER = 1.4
FOCUS_DIST = (0.8, 1.7, 3.0, 4.7, 8.0)
HEIGHT, WIDTH = 480, 640
MIN_DEPTH, MAX_DEPTH = 0.01, 1000.0


class InfinigenLoader(Dataset):
    def __init__(self, root: str):
        self.scenes = sorted(p for p in Path(root).iterdir() if p.is_dir())
        self.focus_dist = torch.tensor(FOCUS_DIST)

    def __len__(self):
        return len(self.scenes)

    def readImage(self, path):
        return np.asarray(Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS))

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        stack = np.stack([self.readImage(scene / f"image_ap_{FNUMBER:.2f}_fd_{fd:.2f}.png") for fd in FOCUS_DIST])
        aif = allInFocus(stack)

        depth = np.load(scene / "depth.npy").astype(np.float32)
        depth = np.asarray(Image.fromarray(depth).resize((WIDTH, HEIGHT), Image.NEAREST))
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
