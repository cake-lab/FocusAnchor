import torch
from torch.utils.data import Dataset
from dataloader.NYU.labeled import NYUv2LabeledDataset
from typing import Tuple


class NYULoader(Dataset):
    """A PyTorch dataset for depth from focus (DFF)."""

    MIN_DEPTH_METERS = 1e-3
    MAX_DEPTH_METERS = 10.0

    def __init__(
        self,
        nyuv2_data_root: str = "",
        img_size: Tuple = (480, 640),
        remove_white_border: bool = True,
        stage: str = "test",
        n_stack: int = 10,
    ) -> None:
        super(NYULoader, self).__init__()

        self.nyuv2_dataset = NYUv2LabeledDataset(
            nyuv2_data_root,
            img_size=img_size,
            remove_white_border=remove_white_border,
            stage=stage,
        )
        self.n_stack = n_stack

    def __len__(self) -> int:
        return len(self.nyuv2_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rgb_aif, depth = self.nyuv2_dataset[index]
        mask = (depth > self.MIN_DEPTH_METERS) & (depth < self.MAX_DEPTH_METERS)
        if not torch.any(mask).item():
            raise ValueError("NYU sample has no valid depth values.")

        # The stack spans the depth actually present in the sample, so no slice
        # is focused on nothing.
        valid_depth = depth[mask]
        return {
            "rgb_aif": rgb_aif / 255.0,
            "depth": depth,
            "focus_dist": torch.linspace(
                torch.min(valid_depth), torch.max(valid_depth), steps=self.n_stack
            ),
            "mask": mask,
            "synthetic_camera": "nyu",
        }
