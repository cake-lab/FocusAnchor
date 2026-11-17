import h5py
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py


class DDFF12Loader(Dataset):
    MAX_STACK_SIZE = 10
    INPUT_MEAN = [0.485, 0.456, 0.406]
    INPUT_STD = [0.229, 0.224, 0.225]

    def __init__(self, hdf5_filename,  stack_key="stack_train", disp_key="disp_train", transform=None,
                 n_stack=10, min_disp=0.02, max_disp=0.28, b_test=False,
                 augment: bool | None = None):
        """
        Args:
            root_dir_fs (string): Directory with all focal stacks of all image datasets.
            root_dir_depth (string): Directory with all depth images of all image datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Opened lazily, once per process. A handle created here would be inherited
        # by forked DataLoader workers, and sharing one across processes corrupts
        # concurrent reads.
        self.hdf5_filename = hdf5_filename
        self.hdf5 = None
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.augment = 'train' in stack_key if augment is None else augment
        if not 1 <= n_stack <= self.MAX_STACK_SIZE:
            raise ValueError("DDFF12 n_stack must be between 1 and 10.")
        # "stack_val" -> "AiF_val": the h5 ships a real all-in-focus image per sample.
        self.aif_key = stack_key.replace("stack", "AiF")
        with h5py.File(hdf5_filename, 'r') as h5:
            self.length = h5[stack_key].shape[0]
        self.max_n_stack = self.MAX_STACK_SIZE
        self.b_test = b_test

        self.n_stack = n_stack
        self.disp_dist = torch.linspace(
            max_disp, min_disp, steps=self.max_n_stack)
        self.focus_dists_max = self.disp_dist.max()
        self.focus_dists_min = self.disp_dist.min()

        if transform is None:
            if self.augment:
                self.transform = self.__create_preprocessing(crop_size=224, cliprange=None, b_filp=True)
            else:
                self.transform = torchvision.transforms.Compose([
                    DDFF12Loader.ToTensor(),
                    DDFF12Loader.PadSamples((384, 576)),
                    DDFF12Loader.Normalize(mean_input=self.INPUT_MEAN,
                                           std_input=self.INPUT_STD),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self._read_sample(idx)

        # Transform sample with data augmentation transformers
        sample_out = self.transform(sample) if self.transform else sample

        # _read_sample appends the all-in-focus image as the last frame, so the
        # transforms hit it identically. Split it back off before sub-stacking.
        stack_imgs = sample_out['input'][:-1]
        rgb_aif = self._denormalize_rgb(sample_out['input'][-1])

        if self.augment and self.n_stack < self.max_n_stack:
            indices = np.sort(np.random.choice(
                self.max_n_stack, self.n_stack, replace=False))
        else:
            indices = np.linspace(0, self.max_n_stack - 1, self.n_stack).astype(np.int64)

        out_imgs = stack_imgs[indices]
        out_disp = sample_out['output']
        focus_dists = self.disp_dist[indices]

        # mask is any pixel with depth greater than 0.02 and less than 0.28
        mask = (out_disp > self.focus_dists_min) & (out_disp < self.focus_dists_max)

        return rgb_aif, out_imgs, out_disp, focus_dists, mask

    def _read_sample(self, idx):
        if self.hdf5 is None:
            self.hdf5 = h5py.File(self.hdf5_filename, 'r')

        # float32, not float: the stack is stored as uint8 and ToTensor casts down to
        # float32 anyway, so float64 here is 8x the memory traffic. The all-in-focus
        # image rides along as the last frame so it gets the same flip/crop/pad/norm
        # as the stack; __getitem__ splits it off again.
        stack = self.hdf5[self.stack_key][idx].astype(np.float32)
        aif = self.hdf5[self.aif_key][idx][None].astype(np.float32)
        inputs = np.concatenate([stack, aif], axis=0)

        if 'test' in self.stack_key:
            return {'input': inputs, 'output': np.ones([2, 2])}

        return {'input': inputs, 'output': self.hdf5[self.disp_key][idx]}

    def _denormalize_rgb(self, rgb):
        mean = torch.tensor(self.INPUT_MEAN, device=rgb.device).view(3, 1, 1)
        std = torch.tensor(self.INPUT_STD, device=rgb.device).view(3, 1, 1)
        return rgb * std + mean

    def __create_preprocessing(self, crop_size=None, cliprange=None, b_filp=True,  mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]):
        # The real data mean and std
        # mean [0.40134945 0.48795037 0.45252803], std [0.14938343, 0.15709994, 0.1496986]
        if b_filp:
            transform = [self.RandomFilp(), self.ToTensor()]
        else:
            transform = [self.ToTensor()]

        if cliprange is not None:
            transform += [self.ClipGroundTruth(cliprange[0], cliprange[1])]
        if crop_size is not None:
            transform += [self.RandomCrop(crop_size)]
        if crop_size is None:
            transform += [self.PadSamples((384, 576))]
        if mean is not None and std is not None:
            transform += [self.Normalize(mean_input=mean, std_input=std)]
        transform = torchvision.transforms.Compose(transform)
        return transform

    def get_stack_size(self):
        return self.__getitem__(0)[1].shape[0]

    class Normalize(object):
        def __init__(self, mean_input, std_input, mean_output=None, std_output=None):
            self.mean_input = mean_input
            self.std_input = std_input
            self.mean_output = mean_output
            self.std_output = std_output

        def __call__(self, sample):
            img_lst = []
            samples = sample['input']

            for i, sample_input in enumerate(samples):
                img_lst.append(torchvision.transforms.functional.normalize(
                    sample_input, mean=self.mean_input, std=self.std_input))
            input_images = torch.stack(img_lst)

            if self.mean_output is None or self.std_output is None:
                output_image = sample['output']
            else:
                output_image = torchvision.transforms.functional.normalize(sample['output'], mean=self.mean_output,
                                                                           std=self.std_output)

            return {'input': input_images, 'output': output_image}

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):
            # Add color dimension to depth map
            sample['output'] = np.expand_dims(sample['output'], axis=0)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            sample['input'] = torch.from_numpy(sample['input'].transpose(
                (0, 3, 1, 2))).float().div(255)
            sample['output'] = torch.from_numpy(sample['output']).float()
            return sample

    class ClipGroundTruth(object):
        def __init__(self, lower_bound, upper_bound):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def __call__(self, sample):
            sample['output'][sample['output'] < self.lower_bound] = 0.0
            sample['output'][sample['output'] > self.upper_bound] = 0.0
            return sample

    class RandomFilp(object):
        """ Randomly crop images
        """

        def __init__(self, ratio=0.5):
            self.ratio = ratio

        def __call__(self, sample):
            inputs, target = sample['input'], sample['output']

            # hori filp
            if np.random.binomial(1, self.ratio):
                inputs = inputs[:, :, ::-1]
                target = target[:, ::-1]

            # vert flip
            if np.random.binomial(1, self.ratio):
                inputs = inputs[:, ::-1]
                target = target[::-1]

            return {'input': np.ascontiguousarray(inputs), 'output': np.ascontiguousarray(target)}

    class RandomCrop(object):
        def __init__(self, output_size, valid_crop_threshold=0.8):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            self.valid_crop_threshold = valid_crop_threshold

        def __is_valid_crop(self, output_image, valid_pixel_cond=lambda x: x >= 0.01):
            valid_occurrances = valid_pixel_cond(output_image).sum()
            all_occurances = np.prod(output_image.shape)
            return (float(valid_occurrances) / float(all_occurances)) >= self.valid_crop_threshold

        def __call__(self, sample):
            h, w = sample['input'].shape[2:4]
            new_h, new_w = self.output_size

            # Draw positions and keep the first valid one. Enumerating every legal
            # position (52k on DDFF12) and shuffling the whole array cost ~31 ms per
            # sample, half the loading time, to pick one of them. If 32 draws all
            # miss, the last one stands, as the exhaustive version also fell back to
            # an unchecked crop.
            for _ in range(32):
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
                output_image = sample['output'][:, top:(top + new_h), left:(left + new_w)]
                if self.__is_valid_crop(output_image):
                    break

            input_images = sample['input'][:, :, top:(top + new_h), left:(left + new_w)]
            return {'input': input_images, 'output': output_image}

    class PadSamples(object):
        def __init__(self, output_size, ground_truth_pad_value=0.0):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            self.ground_truth_pad_value = ground_truth_pad_value

        def __call__(self, sample):
            h, w = sample['input'].shape[2:4]
            new_h, new_w = self.output_size
            padh = np.int32(new_h - h)
            padw = np.int32(new_w - w)
            sample['input'] = torch.stack(
                [torch.from_numpy(np.pad(sample_input.numpy(), ((0, 0), (0, padh), (0, padw)), mode="reflect")).float()
                 for sample_input in sample['input']])
            sample['output'] = torch.from_numpy(
                np.pad(sample['output'].numpy(), ((0, 0), (0, padh), (0, padw)), mode="constant",
                       constant_values=self.ground_truth_pad_value)).float()

            return sample

    class RandomSubStack(object):
        def __init__(self, output_size):
            self.output_size = output_size

        def __call__(self, sample):
            sample['input'] = torch.stack([sample['input'][i] for i in
                                           np.random.choice(sample['input'].shape[0], self.output_size, replace=False)])
            return sample
