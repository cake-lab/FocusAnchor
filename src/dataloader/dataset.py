from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from dataloader.DDFF12.loader import DDFF12Loader
from dataloader.Infinigen.loader import InfinigenLoader
from dataloader.NYU.loader import NYULoader
from dataloader.RGBD.loader import RGBDLoader
from dataloader.ZEDD.loader import ZEDDLoader


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, shuffle_train: bool) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

    def makeLoader(
        self,
        dataset: Dataset,
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
        collate_fn=None,
        sampler=None,
    ) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


class DDFF12DataModule(BaseDataModule):
    def __init__(
        self,
        ddff12_data_root: str = "",
        batch_size: int = 32,
        num_workers: int = 16,
        shuffle_train: bool = True,
        val_fraction: float = 0.0,
        split_seed: int = 0,
        n_stack: int = 5,
        train_n_stack: tuple[int, int] | None = None,
    ):
        """Optionally reserve training images for development, leaving stack_val for test.

        `n_stack` slices are used everywhere, unless `train_n_stack` = (min, max) is set.
        Then each training batch draws its own stack size in that range, and each image
        its own random slices. Validation and test keep `n_stack`."""
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=shuffle_train,
        )
        if not 0 <= val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1).")
        if train_n_stack and not 1 <= train_n_stack[0] <= train_n_stack[1] <= 10:
            raise ValueError("train_n_stack must be (min, max) with 1 <= min <= max <= 10.")
        self.database_root = ddff12_data_root
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.n_stack = n_stack
        self.train_n_stack = train_n_stack
        self.train_indices = []
        self.val_indices = []

    def setup(self, stage: str | None = None) -> None:
        self.stage = stage
        # The official validation split remains the test set for every mode.
        self.test_dataset = self.buildDataset("stack_val", "disp_val", self.n_stack, b_test=True)
        self.valid_dataset = self.test_dataset

        # With a random stack size, training loads all 10 slices and the collate picks.
        trainStack = 10 if self.train_n_stack else self.n_stack
        if stage in {None, "fit", "validate"} and self.val_fraction > 0:
            train = self.buildDataset("stack_train", "disp_train", trainStack)
            n_val = int(len(train) * self.val_fraction)
            if not 0 < n_val < len(train):
                raise ValueError("val_fraction must leave nonempty training and validation splits.")
            generator = torch.Generator().manual_seed(self.split_seed)
            indices = torch.randperm(len(train), generator=generator).tolist()
            self.val_indices = sorted(indices[:n_val])
            self.train_indices = sorted(indices[n_val:])
            self.train_dataset = Subset(train, self.train_indices)
            self.valid_dataset = Subset(
                self.buildDataset("stack_train", "disp_train", self.n_stack, augment=False),
                self.val_indices,
            )
        elif stage in {None, "fit"}:
            self.train_dataset = self.buildDataset("stack_train", "disp_train", trainStack)

    def state_dict(self):
        # Exact image membership travels with the checkpoint.
        return {
            "val_fraction": self.val_fraction,
            "split_seed": self.split_seed,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
        }

    def load_state_dict(self, state_dict):
        # setup runs before checkpoint restoration; verify its deterministic split.
        # A standalone test has nothing to check.
        if getattr(self, "stage", None) not in {None, "fit", "validate"}:
            return
        for key in ["val_fraction", "split_seed", "train_indices", "val_indices"]:
            if key in state_dict and state_dict[key] != getattr(self, key):
                raise ValueError(f"DDFF12 checkpoint {key} does not match the current data configuration.")

    def train_dataloader(self):
        return self.makeLoader(self.train_dataset, shuffle=self.shuffle_train,
                               collate_fn=self.randomStackSize if self.train_n_stack else None)

    def randomStackSize(self, samples):
        """One random stack size per batch, so the batch still stacks, then random
        slices per image, sorted near to far."""
        n = np.random.randint(self.train_n_stack[0], self.train_n_stack[1] + 1)
        out = []
        for aif, stack, disp, focus, mask in samples:
            idx = np.sort(np.random.choice(len(stack), n, replace=False))
            out.append((aif, stack[idx], disp, focus[idx], mask))
        return default_collate(out)

    def val_dataloader(self):
        # Full batches: validation runs under no_grad, and calcMetrics averages per
        # image, so batching changes the speed and not the numbers. Test stays at 1
        # because its `seconds` metric is a per-batch timing.
        return self.makeLoader(self.valid_dataset)

    def test_dataloader(self):
        return self.makeLoader(self.test_dataset, batch_size=1)

    def buildDataset(
        self,
        stack_key: str,
        disp_key: str,
        n_stack: int,
        *,
        b_test: bool = False,
        augment: bool | None = None,
    ) -> DDFF12Loader:
        return DDFF12Loader(
            self.database_root,
            stack_key=stack_key,
            disp_key=disp_key,
            n_stack=n_stack,
            min_disp=0.02,
            max_disp=0.28,
            b_test=b_test,
            augment=augment,
        )


class NYUDataModule(BaseDataModule):
    def __init__(
        self,
        nyuv2_data_root: str = "",
        img_size: Tuple = (480, 640),
        remove_white_border: bool = True,
        batch_size: int = 32,
        num_workers: int = 16,
        n_stack: int = 10,
        shuffle_train: bool = True,
    ):
        """Initialize the data module."""
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=shuffle_train,
        )
        self.nyu_data_root = nyuv2_data_root
        self.image_size = img_size
        self.remove_white_border = remove_white_border
        self.n_stack = n_stack

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = self.buildDataset("train")

        self.test_dataset = self.buildDataset("test")

    def train_dataloader(self):
        return self.makeLoader(self.train_dataset, shuffle=self.shuffle_train)

    def val_dataloader(self):
        # See DDFF12DataModule.val_dataloader on why this is not batch_size=1.
        return self.makeLoader(self.test_dataset)

    def test_dataloader(self):
        return self.makeLoader(self.test_dataset, batch_size=1)

    def buildDataset(self, stage: str) -> NYULoader:
        return NYULoader(
            self.nyu_data_root,
            img_size=self.image_size,
            remove_white_border=self.remove_white_border,
            n_stack=self.n_stack,
            stage=stage,
        )


class ZEDDDataModule(BaseDataModule):
    """ZEDD val split, test only. Zero-shot: nothing trains on ZEDD."""

    def __init__(self, zedd_root: str = "", aif: str = "fused", num_workers: int = 4):
        super().__init__(batch_size=1, num_workers=num_workers, shuffle_train=False)
        self.zedd_root = zedd_root
        self.aif = aif

    def setup(self, stage: str | None = None) -> None:
        self.test_dataset = ZEDDLoader(self.zedd_root, aif=self.aif)

    def test_dataloader(self):
        return self.makeLoader(self.test_dataset)


class InfinigenDataModule(BaseDataModule):
    """Infinigen Defocus, test only. Zero-shot: nothing trains on Infinigen."""

    def __init__(self, infinigen_root: str = "", num_workers: int = 4):
        super().__init__(batch_size=1, num_workers=num_workers, shuffle_train=False)
        self.infinigen_root = infinigen_root

    def setup(self, stage: str | None = None) -> None:
        self.test_dataset = InfinigenLoader(self.infinigen_root)

    def test_dataloader(self):
        return self.makeLoader(self.test_dataset)


class RGBDDataModule(BaseDataModule):
    """iBims-1, HAMMER or ARKitScenes with FOSSA's rendered stacks, test only. Zero-shot. The loader
    renders on the GPU, so num_workers stays 0."""

    def __init__(self, root: str = "", dataset: str = "ibims", num_workers: int = 0):
        super().__init__(batch_size=1, num_workers=num_workers, shuffle_train=False)
        self.root = root
        self.dataset = dataset

    def setup(self, stage: str | None = None) -> None:
        self.test_dataset = RGBDLoader(self.root, self.dataset)

    def test_dataloader(self):
        return self.makeLoader(self.test_dataset)
