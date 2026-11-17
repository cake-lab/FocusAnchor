"""A loader for the labeled ARKitScenes dataset."""
import os

import imageio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2


class ARKitscenesDataset(Dataset):
    """Python interface for the labeled subset of the ARKitScenes dataset."""

    def __init__(self, path, img_size=(192, 256), stage="Validation"):
        self.root_dir = path
        self.stage = stage
        split_path = os.path.join(
            os.path.dirname(__file__), "input_splits", f"{stage}.txt"
        )
        with open(split_path, "r") as f:
            self.samples_pth = f.readlines()
        self.img_size = img_size

    def __len__(self):
        return len(self.samples_pth)

    def __getitem__(self, idx):
        sample_path = self.samples_pth[idx].split()[0]
        color_path = os.path.join(
            self.root_dir, self.stage, sample_path.replace("lowres_wide", "wide")
        )
        depth_path = os.path.join(self.root_dir, self.stage, sample_path).replace(
            "lowres_wide", "highres_depth"
        )

        color_img = imageio.imread(color_path, pilmode="RGB")
        depth_img = imageio.imread(depth_path, pilmode="I")

        color_img = torch.from_numpy(color_img).float()
        depth_img = torch.from_numpy(depth_img).float()
        depth_img = depth_img / 1000.0
        depth_img = depth_img.unsqueeze(0)

        color_img = color_img.permute(2, 0, 1)
        color_img = transforms_v2.Resize(self.img_size, antialias=True)(color_img)
        # Nearest, not bilinear: bilinear blends hole zeros into fake small depths at
        # hole edges, and the nearest focus plane then lands near 0 m.
        depth_img = transforms_v2.Resize(
            self.img_size, interpolation=transforms_v2.InterpolationMode.NEAREST)(depth_img)

        return color_img, depth_img
