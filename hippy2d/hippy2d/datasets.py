from abc import ABC
import os

from lightning import LightningDataModule
from loguru import logger
import torch
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from typing import Optional, Callable, Tuple, Any, Dict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import InterpolationMode
from hippy2d.utils import get_optimal_workers, get_default_complex,tukey_2d
from hippy2d.benchmarks.mnist_rot import build_mnist_rot_loader
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchvision.datasets.utils import (
    download_and_extract_archive,
    verify_str_arg,
)
from torchvision.datasets import ImageFolder
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Custom Transforms
class NormalizeMagnitude(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        assert std != 0, "Standard deviation cannot be zero"
        self.mean = mean
        self.std = std

    def forward(self, img, eps=1e-6):
        # Do some transformations
        assert img.dtype == get_default_complex(), "Input must be complex"
        magnitude = img.abs()
        normalized_magnitude = (img.abs() - self.mean) / self.std
        norm = normalized_magnitude / torch.clamp(magnitude, min=eps)
        return norm * img


class MnistRotTestDataset(VisionDataset):
    # TODO: Connect to hugging faces URL and MD5Sum
    """
    mnist-rot-test pytorch faster reimplementation of ../custom_datasets.py hugging faces dataset
    Based on:
    https://pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html#MNIST
    """
    _VERSION = "v3"
    _URL = "https://owncloud.cesnet.cz/index.php/s/q2BYzg8Uzcc8O4g/download"
    _MD5SUM = "f1789ad4651263d3667a17fb651acc95"
    _FILE_NAME = f"mnist-rot-test-uint8-{_VERSION}.npz"

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        logger.warning("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        logger.warning("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        logger.warning("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        logger.warning("test_data has been renamed data")
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def __init__(self,
                 root: str,
                 train: Optional[bool] = None,
                 split: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True):
        self.processed_file = None
        assert (train is None) or (split is None), "Use only one parameter split or train"
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform)

        if train is None:
            assert split in ['train', 'test'], "Supporting two options for split. (train, test)"
            self.train = (split == 'train')
            self.split = split
        else:
            self.train = train
            self.split = 'train' if train else 'test'

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_data(self):
        with np.load(self._downloaded_file) as file:
            data = torch.from_numpy(file[f"{self.split}_images"])
            targets = torch.from_numpy(file[f"{self.split}_labels"])
        return data, targets

    @property
    def _downloaded_file(self):
        return os.path.join(self.root, self._FILE_NAME)

    def _check_exists(self) -> bool:
        return check_integrity(self._downloaded_file, md5=self._MD5SUM)

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return
        logger.debug(f"Downloading {self._URL}")
        download_url(self._URL,
                     root=self.root,
                     filename=self._FILE_NAME,
                     md5=self._MD5SUM)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class MnistRotTest(LightningDataModule, ABC):
    def __init__(self,
                 data_dir: str = "./data",
                 pad: int = 0,
                 batch_size: int = 32,
                 test_batch_size: int = 256,
                 scale_factor: int = 2,
                 scale_mode="BILINEAR",
                 normalize=False,
                 limit_train_samples=None,
                 to_complex=True,
                 num_workers=None):
        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        assert scale_factor >= 1, f"Scale factor must be higher or equal than one"
        if num_workers is None:
            num_workers = get_optimal_workers()
        super().__init__()
        assert hasattr(InterpolationMode, scale_mode)
        scale_mode = getattr(InterpolationMode, scale_mode)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.valid_ds = None  # Multiple checking multiple angles
        self.test_ds = None
        self.train_ds = None
        self.num_classes = len(MnistRotTestDataset.classes)

        # Splitting indices
        indices = list(range(60000))
        self.valid_indices = indices[:10000]
        self.train_indices = indices[10000:]

        # Limit train samples
        if limit_train_samples is not None:
            assert limit_train_samples <= len(
                self.train_indices), f"Limit is higher than train size. {limit_train_samples}"
            assert limit_train_samples > 0, f"Limit must be higher than zero."
            self.train_indices = self.train_indices[:limit_train_samples]

        self._output_shape = [batch_size, 1, 32 + 2 * pad, 32 + 2 * pad]
        # Transformations
        pre_transforms = [
            transforms.ToImage(),
            transforms.Pad(padding=pad, fill=0, padding_mode='constant'),
        ]

        post_transforms = []
        post_train_transforms = []

        post_train_transforms.append(transforms.ToDtype(torch.get_default_dtype(), scale=True))
        post_transforms.append(transforms.ToDtype(torch.get_default_dtype(), scale=True))

        if scale_factor != 1:
            self.output_shape[2] *= scale_factor
            self.output_shape[3] *= scale_factor
            post_transforms.append(transforms.Resize(
                size=self.output_shape[2],
                interpolation=scale_mode,
                antialias=False))
            post_train_transforms.append(transforms.Resize(
                size=self.output_shape[2],
                interpolation=scale_mode,
                antialias=False))
        if normalize:
            post_transforms.append(transforms.Normalize(mean=[0.1307], std=[0.3081]))
            post_train_transforms.append(transforms.Normalize(mean=[0.1307], std=[0.3081]))
        if to_complex:
            post_train_transforms.append(
                transforms.ToDtype(dtype=get_default_complex())
            )
            post_transforms.append(
                transforms.ToDtype(dtype=get_default_complex())
            )
        self.train_transforms = transforms.Compose(pre_transforms + post_train_transforms)
        self.base_transforms = transforms.Compose(pre_transforms + post_transforms)
        self.rotation_transforms = {
            '0': self.base_transforms,
            '90': transforms.Compose(
                [*pre_transforms,
                 transforms.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),
                 *post_transforms]
            ),
            '45': transforms.Compose(
                [*pre_transforms,
                 transforms.RandomRotation(degrees=(45, 45), interpolation=InterpolationMode.BILINEAR),
                 *post_transforms]),
            'rd': transforms.Compose(
                [*pre_transforms,
                 transforms.RandomRotation(degrees=(0, 359), interpolation=InterpolationMode.BILINEAR),
                 *post_transforms]
            )
        }

        self.num_workers = num_workers

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def _dataset(self):
        return MnistRotTestDataset

    def prepare_data(self):
        self._dataset(root=self.data_dir,
                      transform=self.train_transforms,
                      train=True,
                      download=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True)

    def setup(self, stage: Optional[str]):
        self.train_ds = self._dataset(root=self.data_dir,
                                      split='train',
                                      transform=self.train_transforms)
        # Test/Valid datasets
        self.valid_ds = {}
        for k, _transforms in self.rotation_transforms.items():
            self.valid_ds[k] = self._dataset(root=self.data_dir,
                                             split='train',
                                             transform=_transforms)
        self.test_ds = self._dataset(root=self.data_dir,
                                     split='test',
                                     transform=self.base_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          sampler=RandomSampler(self.train_indices),
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        loaders = {}
        for k, ds in self.valid_ds.items():
            ds_name = f"val_{k}" if k != '0' else 'val'
            loaders[ds_name] = DataLoader(ds,
                                          sampler=SequentialSampler(self.valid_indices),
                                          batch_size=self.test_batch_size,
                                          pin_memory=True,
                                          num_workers=self.num_workers,
                                          persistent_workers=True),
        return CombinedLoader(loaders, mode="max_size_cycle")


class RotMnist(LightningDataModule, ABC):

    @property
    def num_classes(self):
        return 10

    @property
    def output_shape(self):
        return self._output_shape

    def __init__(self,
                data_dir:str='./data', 
                num_workers=8,
                batch_size:int=64,
                interpolation:str="BILINEAR", 
                test_batch_size:int=32,
                augmentation: bool=True,
                ):
        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        assert hasattr(Resampling, interpolation)
        super().__init__()
        self.interpolation = getattr(Resampling, interpolation)
        self.save_hyperparameters(ignore=['data_dir',
                                          'num_workers',
                                          'test_batch_size'])
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.augmentation=augmentation
        self.reshuffle_seed = np.random.randint(0, 100000)

        self._output_shape = [batch_size, 1, 28, 28]


    def prepare_data(self):
        build_mnist_rot_loader(mode='trainval', batch_size=self.batch_size)
        build_mnist_rot_loader(mode='test', batch_size=self.test_batch_size)

    def test_dataloader(self):
        return self._test_dataloader

    def setup(self, stage: Optional[str]):
        self._train_dataloader, _, _ = build_mnist_rot_loader(mode='train',
                                               num_workers=self.num_workers,
                                               batch_size=self.batch_size,
                                               rot_interpol_augmentation=True,
                                               interpolation=self.interpolation,
                                               reshuffle_seed=self.reshuffle_seed,
                                               coords=False                                               )
        self._valid_dataloader = dict(
            val=build_mnist_rot_loader(mode='valid',
                                       num_workers=self.num_workers,
                                       batch_size=self.batch_size,
                                       rot_interpol_augmentation=False,
                                       interpolation=self.interpolation,
                                       reshuffle_seed=self.reshuffle_seed,
                                       coords=False)[0]
        )
        self._test_dataloader, _, _ = build_mnist_rot_loader(mode='test',
                                              num_workers=self.num_workers,
                                              batch_size=self.test_batch_size,
                                              rot_interpol_augmentation=False,
                                              interpolation=self.interpolation,
                                              coords=False)
        

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return CombinedLoader(self._valid_dataloader, mode="max_size_cycle")


class RESISC45(LightningDataModule):
    _MD5SUM = "944a6a08ea97d50609df18ed1d8675b6" 
    _URL = "https://owncloud.cesnet.cz/index.php/s/OofMmlJ6b84HH2l/download"
    _FILE_NAME = f"NWPU-RESISC45"
    num_classes = 45
    total_samples = 31500
    def __init__(self,
                 data_dir: str = "./data",
                 pad: int = 0,
                 batch_size: int = 32,
                 test_batch_size: int = 256,
                 to_complex=False,
                 num_workers=None):
        super().__init__()
        if num_workers is None:
            num_workers = get_optimal_workers()
        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        if not self._check_exists():
            self.download()

        self.train_transforms = [
            transforms.ToImage(),
            #transforms.RandomCrop((224, 224)),
            # TODO: Setup the split 
            transforms.CenterCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToDtype(torch.get_default_dtype(), scale=True)

        ]

        self.valid_transforms = [
            transforms.ToImage(),
            transforms.CenterCrop((224, 224)),
            transforms.ToDtype(torch.get_default_dtype(), scale=True)
        ]

        if to_complex:
            self.valid_transforms.append(
                transforms.ToDtype(dtype=get_default_complex())
            )
            self.train_transforms.append(
                transforms.ToDtype(dtype=get_default_complex())
            )

        self.train_transforms = transforms.Compose(self.train_transforms)
        self.valid_transforms = transforms.Compose(self.valid_transforms)

        self.valid_ds = None  # Multiple checking multiple angles
        self.test_ds = None
        self.train_ds = None
        self.output_shape = [batch_size, 3, 128, 128]
        self.num_workers = num_workers
        

        # Prepare test, validation, and training datasets
        self.test_size = int(0.2 * RESISC45.total_samples)
        self.train_val_size = RESISC45.total_samples - self.test_size
        self.train_size = int(0.8 * self.train_val_size)
        self.val_size = self.train_val_size - self.train_size
        
    def prepare_data(self):
        # Download 
        self.ds = ImageFolder(root=os.path.join(self.data_dir, RESISC45._FILE_NAME),
                              transform=self.transforms)

    def setup(self, stage:str):
        # TODO: Fix the indicies
        ds_test, ds_train_val = torch.utils.data.random_split(self.ds, [self.test_size, self.train_val_size])
        ds_train, ds_val = torch.utils.data.random_split(ds_train_val, [self.train_size, self.val_size])

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        
    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.ds_test, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    @property
    def _downloaded_file(self):
        return os.path.join(self.data_dir, self._FILE_NAME)

    def _check_exists(self) -> bool:
        return check_integrity(f"{self._downloaded_file}.zip", md5=self._MD5SUM)

    def download(self) -> None:
        """Download the RESISC45 data if it doesn't exist already."""

        if self._check_exists():
            return
        logger.debug(f"Downloading {self._URL}")
        download_and_extract_archive(self._URL,
                                     download_root=self.data_dir,
                                     filename=f"{self._FILE_NAME}.zip",
                                     md5=self._MD5SUM)

# Transformations
class CircularPad:
    """ Circular padding transform for 2D images"""

    def __init__(self, size):
        xx, yy = np.meshgrid(np.linspace(-1, 1, num=size),
                             np.linspace(-1, 1, num=size))

        r = np.sqrt(xx ** 2 + yy ** 2)
        self.circle_size = size
        self.circular_mask = torch.from_numpy(r <= 1)

    def __call__(self, x: torch.Tensor):
        assert x.ndim == 3
        assert x.shape[1] == self.circle_size
        assert x.shape[2] == self.circle_size
        return x * self.circular_mask[None, :, :]


class TukeyMask:
    def __init__(self, size, alpha=0.4):
        self.size = size
        self.alpha = alpha
        self.tukey_mask = torch.from_numpy(tukey_2d(size, alpha)).type(torch.get_default_dtype())

    def __call__(self, x: torch.Tensor):
        assert x.ndim == 3
        assert x.shape[1] == self.size
        assert x.shape[2] == self.size
        return x * self.tukey_mask[None, :, :]
