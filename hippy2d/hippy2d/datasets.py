from abc import ABC
import os

from lightning import LightningDataModule
from loguru import logger
import torch
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple, Any, Dict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import InterpolationMode
from hippy2d.utils import get_optimal_workers, get_default_complex,tukey_2d
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
        return self._dataset.num_classes

    @property
    def output_shape(self):
        return self._output_shape

    # TODO: Add normalization
    # TODO: Add masking
    # other transforms
    TRAIN_SIZE = 10000
    TEST_SIZE = 50000
    VAL_SIZE = 2000

    def __init__(self,
                 data_dir: str = "./data",
                 pad: int = 2,
                 batch_size: int = 32,
                 test_batch_size: int = 256,
                 scale_factor: int = 2,
                 scale_mode="BILINEAR",
                 limit_train_samples=None,
                 normalize=True,
                 to_complex=True,
                 num_workers=get_optimal_workers()):
        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        assert scale_factor >= 1, f"Scale factor must be higher or equal than one"
        assert hasattr(InterpolationMode, scale_mode)
        super().__init__()
        scale_mode = getattr(InterpolationMode, scale_mode)
        self.save_hyperparameters(ignore=['input_shape',
                                          'data_dir',
                                          'num_workers',
                                          'input_shape',
                                          'test_batch_size'])
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.valid_ds = None  # Multiple checking multiple angles
        self.test_ds = None
        self.train_ds = None
        self.train_indices = list(range(RotMnist.TRAIN_SIZE))
        self.valid_indices = list(range(RotMnist.VAL_SIZE))
        self.test_indices = list(range(RotMnist.TEST_SIZE))

        # Limit train samples
        if limit_train_samples is not None:
            assert limit_train_samples <= len(
                self.train_indices), f"Limit is higher than train size. {limit_train_samples}"
            assert limit_train_samples > 0, f"Limit must be higher than zero."
            self.train_indices = self.train_indices[:limit_train_samples]

        self._output_shape = [batch_size, 1, 28 + 2 * pad, 28 + 2 * pad]

        rotmnist_transforms = [
            transforms.ToImage(),
            transforms.Pad(padding=pad, fill=0, padding_mode='constant'),
            transforms.ToDtype(torch.get_default_dtype(), scale=True),
        ]

        if scale_factor != 1:
            self.output_shape[2] *= scale_factor
            self.output_shape[3] *= scale_factor
            rotmnist_transforms.append(transforms.Resize(
                size=self.output_shape[2],
                interpolation=scale_mode,
                antialias=False))
        # Taken from ./notebooks/15_rotated_mnist
        rotmnist_transforms.append(transforms.Normalize(mean=[0.0998], std=[0.1770]))

        if to_complex:
            rotmnist_transforms.append(
                transforms.ToDtype(dtype=get_default_complex())
            )
        # TODO: Add masking

        self.train_transforms = transforms.Compose(rotmnist_transforms)
        self.test_transforms = transforms.Compose(rotmnist_transforms)
        self.val_transforms = {
            'val': transforms.Compose(rotmnist_transforms)
        }

    @property
    def _dataset(self):
        return rotMnistDataset

    def prepare_data(self):
        self._dataset(root=self.data_dir,
                      split='train',
                      transform=self.train_transforms,
                      download=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=False)

    def setup(self, stage: Optional[str]):
        self.train_ds = self._dataset(root=self.data_dir,
                                      split='train',
                                      transform=self.train_transforms)
        # Test/Valid datasets
        self.valid_ds = {}
        for k, _transforms in self.val_transforms.items():
            self.valid_ds[k] = self._dataset(root=self.data_dir,
                                             split='valid',
                                             transform=_transforms)
        self.test_ds = self._dataset(root=self.data_dir,
                                     split='test',
                                     transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          sampler=RandomSampler(self.train_indices),
                          batch_size=self.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        loaders = {}
        for k, ds in self.valid_ds.items():
            ds_name = f"val_{k}" if k != 'val' else 'val'
            loaders[ds_name] = DataLoader(ds,
                                          sampler=SequentialSampler(self.valid_indices),
                                          batch_size=self.test_batch_size,
                                          pin_memory=True,
                                          num_workers=self.num_workers),
        return CombinedLoader(loaders, mode="max_size_cycle")

# NOTE: For reproducibility, this was taken from
# https://github.com/dwromero/g_selfatt/blob/14204b3eb2a9d70329ee8f33a04ac5965c11e8c6/datasets/mnist_rot.py
# NOTE: Unlike the original code we use validation set for validation

class rotMnistDataset(VisionDataset):
    """Rotated MNIST datasets.

    Download the datasets from https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits
    and preprocess it as in (Cohen and Welling) https://github.com/tscohen/gconv_experiments/blob/master/gconv_experiments/MNIST_ROT/mnist_rot.py
    """

    resources = [
        (
            "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip",
            "0f9a947ff3d30e95cd685462cbf3b847",
        ),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    valid_file = "valid.pt"
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
    num_classes = len(classes)

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "test", 'valid'))

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        if self.split == 'train':
            data_file = self.training_file
        elif self.split == 'valid':
            data_file = self.valid_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        # process and save as torch files
        print("Processing...")

        train_filename = os.path.join(
            self.raw_folder, "mnist_all_rotation_normalized_float_train_valid.amat"
        )
        test_filename = os.path.join(
            self.raw_folder, "mnist_all_rotation_normalized_float_test.amat"
        )

        train_val = torch.from_numpy(np.loadtxt(train_filename))
        test = torch.from_numpy(np.loadtxt(test_filename))

        train_val_data = train_val[:, :-1].reshape(-1, 28, 28)
        train_val_data = (train_val_data * 256).round().type(torch.uint8)
        train_val_labels = train_val[:, -1].type(torch.uint8)
        training_set = (train_val_data[:10000], train_val_labels[:10000])
        # Init validation set unlike original code
        validation_set = (train_val_data[10000:], train_val_labels[10000:])

        test_data = test[:, :-1].reshape(-1, 28, 28)
        test_data = (test_data * 256).round().type(torch.uint8)
        test_labels = test[:, -1].type(torch.uint8)
        test_set = (test_data, test_labels)

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)
        with open(os.path.join(self.processed_folder, self.valid_file), "wb") as f:
            torch.save(validation_set, f)

        print("Done!")

    def extra_repr(self):
        return f"Split: {self.split}"


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

        # Transforms
        self.transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.get_default_dtype(), scale=True)
        ]

        if to_complex:
            self.transforms.append(
                transforms.ToDtype(dtype=get_default_complex())
            )
        
        self.transforms = transforms.Compose(self.transforms)
        
        self.valid_ds = None  # Multiple checking multiple angles
        self.test_ds = None
        self.train_ds = None
        self.output_shape = [batch_size, 3, 256, 256]
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
