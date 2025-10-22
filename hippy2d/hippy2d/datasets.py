from abc import ABC
import os

from lightning import LightningDataModule
from loguru import logger
from sklearn.model_selection import train_test_split
import torch
import pickle
import datasets
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from typing import Optional, Callable, Tuple, Any, Dict
from einops import rearrange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, default_collate
from torchvision.datasets import VisionDataset
from torchvision.datasets import EuroSAT as EuroSATTorch
from torchvision.datasets.utils import check_integrity, download_url
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import InterpolationMode
from hippy2d.utils import get_optimal_workers, get_default_complex,tukey_2d
from hippy2d.benchmarks.mnist_rot import build_mnist_rot_loader
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader


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
                upscale_size: Optional[int]=None,
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
        self.upscale_size = upscale_size

        if upscale_size is not None:
            self._output_shape = [batch_size, 1, upscale_size, upscale_size]
        else:
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
                                               coords=False,
                                               upscale_size=self.upscale_size)
        self._valid_dataloader = dict(
            val=build_mnist_rot_loader(mode='valid',
                                       num_workers=self.num_workers,
                                       batch_size=self.batch_size,
                                       rot_interpol_augmentation=False,
                                       interpolation=self.interpolation,
                                       reshuffle_seed=self.reshuffle_seed,
                                       coords=False,
                                       upscale_size=self.upscale_size)[0]
        )
        self._test_dataloader, _, _ = build_mnist_rot_loader(mode='test',
                                              num_workers=self.num_workers,
                                              batch_size=self.test_batch_size,
                                              rot_interpol_augmentation=False,
                                              interpolation=self.interpolation,
                                              coords=False,
                                              upscale_size=self.upscale_size)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return CombinedLoader(self._valid_dataloader, mode="max_size_cycle")

def collate_tuple(batch):
    # Let PyTorch stack dicts first, then return a tuple
    b = default_collate(batch)
    return b["image"], b["label"]

class RESISC45(LightningDataModule):
    # Hugging face bridge
    total_samples = 31500

    @property
    def num_classes(self):
        return 45

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


        self.train_transforms = [
            transforms.ToImage(),
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToDtype(torch.get_default_dtype(), scale=True)

        ]

        self.valid_transforms = [
            transforms.ToImage(),
            transforms.Resize((160, 160)),
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
        self.output_shape = [batch_size, 3, 224, 224]
        self.num_workers = num_workers
        

    def prepare_data(self):
        # Download 
        self.hg_dataset_train = datasets.load_dataset("timm/resisc45", split='train')
        self.hg_dataset_valid = datasets.load_dataset("timm/resisc45", split='validation')
        self.hg_dataset_test = datasets.load_dataset("timm/resisc45", split='test')

    def setup(self, stage:str):
        self.ds_train = self.hg_dataset_train.with_transform(self.train_transforms)
        self.ds_val = self.hg_dataset_valid.with_transform(self.valid_transforms)
        self.ds_test = self.hg_dataset_test.with_transform(self.valid_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_tuple
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_tuple
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.ds_test, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_tuple
        )

class EuroSAT(LightningDataModule):
    total_samples = 27000

    @property
    def num_classes(self):
        return 10
    @property
    def output_shape(self):
        return [self.batch_size, 3, 64, 64]
        
    _MEAN = [0.3444, 0.3803, 0.4078]
    _STD = [0.0914, 0.0651, 0.0552]

    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4, to_complex=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        valid_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=EuroSAT._MEAN, 
                               std=EuroSAT._STD)
        ]
        
        train_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=EuroSAT._MEAN,
                               std=EuroSAT._STD)
        ]
    
        if to_complex:
            valid_transform_list.append(
                transforms.ToDtype(dtype=get_default_complex())
            )
            train_transform_list.append(
                transforms.ToDtype(dtype=get_default_complex())
            )
            
        self.transform = transforms.Compose(valid_transform_list)
        self.train_transform = transforms.Compose(train_transform_list)
    def prepare_data(self):
        # Download dataset
        full_dataset = EuroSATTorch(root=self.data_dir, download=True)
        indices = list(range(len(full_dataset)))
        labels = [full_dataset[i][1] for i in indices]  # class labels
        self.train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
        self.val_idx, self.test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)

    def setup(self, stage: str = None):
        # Load full dataset
        traintest_dataset = EuroSATTorch(root=self.data_dir, transform=self.train_transform)
        valid_dataset = EuroSATTorch(root=self.data_dir, transform=self.transform)

        self.train_dataset = torch.utils.data.Subset(traintest_dataset, self.train_idx)
        self.val_dataset = torch.utils.data.Subset(valid_dataset, self.val_idx)
        self.test_dataset = torch.utils.data.Subset(traintest_dataset, self.test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        

class StrainedNormalize(torch.nn.Module):
    """
    Torch-style wrapper for torchstain normalizers.
    Can either take a fitted normalizer object or a path to a pickled one.
    
    Expects input: np.uint8 RGB image (HxWx3).
    Returns: np.uint8 RGB image (HxWx3).
    """
    def __init__(self, normalizer=None, pickle_path=None):
        super().__init__()
        if normalizer is None and pickle_path is None:
            raise ValueError("Provide either a normalizer or a pickle_path")
        if pickle_path is not None:
            with open(pickle_path, "rb") as f:
                self.normalizer = pickle.load(f)
        else:
            self.normalizer = normalizer

        self.min_tissue_ratio=0.15
        self.sat_thresh=0.10

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        image: torch.uint8, 3xHxW (RGB)
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input must be a torch Tensor")
        assert image.ndim == 3 and image.shape[0] == 3, "Input must be 3xHxW RGB image"
        assert image.dtype == torch.uint8, "Input must be uint8"
        rgb_image = Image.fromarray(rearrange(image, 'c h w -> h w c').cpu().numpy(), mode='RGB')
        hsv = np.array(rgb_image.convert('HSV'))
        s = hsv[..., 1].astype(np.float32) / 255.0
        tr = float((s > self.sat_thresh).mean())
        if tr < self.min_tissue_ratio:
            # Too little tissue -> return original to avoid empty-percentile crash
            return image, label
        image, _, _ = self.normalizer.normalize(image, stains=True)  # input uint8 [3xHxW] 
        # assert the values 0 to 255
        assert image.min() >= 0 and image.max() <= 255, "Output must be in range [0, 255]"
        image = rearrange(image, 'h w c -> c h w').to(torch.uint8) # to HxWx3
        return image, label

class ColorectalHistology(LightningDataModule):
    # Hugging face bridge to https://huggingface.co/datasets/dpdl-benchmark/colorectal_histology
    _MEAN = [0.6497, 0.4718, 0.5838]
    _STD = [0.1412, 0.1451, 0.1271]
    @property
    def num_classes(self):
        return 8

    def __init__(self, 
                 data_dir: str = "./data",
                 batch_size: int = 32,
                 test_batch_size: int = 256,
                 to_complex=False,
                 normalize=False,
                 aug_crop=False,
                 aug_scale=False,
                 aug_clr_jitter=False,
                 num_workers=None):
        super().__init__()
        if num_workers is None:
            num_workers = get_optimal_workers()

        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        
        self.train_transforms = [transforms.ToImage()]
        self.valid_transforms = [transforms.ToImage()]
        if aug_scale:
            self.train_transforms.append(transforms.RandomResize(min_size=128, max_size=170))
        if aug_crop:
            self.train_transforms.append(transforms.RandomCrop((128, 128)))
            self.valid_transforms.append(transforms.CenterCrop((128, 128)))
        if aug_clr_jitter:
            self.train_transforms.append(
                transforms.ColorJitter(                            
                    brightness=0.1,
                    contrast=0.1,
                   saturation=0.1,
                    hue=0.03,
                ))
        self.train_transforms.append(transforms.ToDtype(torch.get_default_dtype(), scale=True))
        self.valid_transforms.append(transforms.ToDtype(torch.get_default_dtype(), scale=True))

        if normalize:
            self.valid_transforms.append(transforms.Normalize(mean=self._MEAN, std=self._STD))
            self.train_transforms.append(transforms.Normalize(mean=self._MEAN, std=self._STD))

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
        if aug_crop:
            self.output_shape = [batch_size, 3, 128, 128]
        else:
            self.output_shape = [batch_size, 3, 150, 150]
        self.num_workers = num_workers

    def prepare_data(self): 
        # Download and prepare the dataset
        self.hg_dataset = datasets.load_dataset("dpdl-benchmark/colorectal_histology", split='train')
        labels = np.array([example['label'] for example in self.hg_dataset])
        self.train_idx, test_valid_idx = train_test_split(np.arange(len(labels)),
                                             test_size=0.2, 
                                             random_state=42,
                                             stratify=labels)
        self.valid_idx, self.test_idx = train_test_split(test_valid_idx,
                                       test_size=0.5,
                                       random_state=42,
                                       stratify=labels[test_valid_idx])
    def setup(self, stage: str):
        self.train_ds = self.hg_dataset.select(self.train_idx).with_transform(self.train_transforms)
        self.valid_ds = self.hg_dataset.select(self.valid_idx).with_transform(self.valid_transforms)
        self.test_ds = self.hg_dataset.select(self.test_idx).with_transform(self.valid_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_tuple
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_tuple
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_tuple
        )

class StrainedColorectalHistologyDataset(VisionDataset):
    _FILE_NAME = f"colorectal_histology_stained.npz"
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        """
        Args:
            root (str): Root directory where the .npz file is located
            split (str): Dataset split to use ('train', 'valid', or 'test')
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, transforms.RandomCrop
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool): If True, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f"Split must be one of ['train', 'valid', 'test'], got {split}")
        
        self.split = split
        self.data_file = os.path.join(root, StrainedColorectalHistologyDataset._FILE_NAME)
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        self._load_data()
    
    def _check_exists(self) -> bool:
        """Check if the .npz file exists."""
        return os.path.exists(self.data_file)
    
    def download(self):
        """
        Download the dataset. This should be implemented based on where the 
        stain-normalized data is hosted.
        """
        if self._check_exists():
            return
        
        raise NotImplementedError(
            "Automatic download not implemented. Please create the stain-normalized "
            f"dataset and save it as '{self.data_file}' using the notebook preprocessing steps."
        )
    
    def _load_data(self):
        """Load data from the .npz file."""
        data = np.load(self.data_file)
        
        # Load appropriate split
        if self.split == 'train':
            self.data = data['train_imgs']
            self.targets = data['train_labels']
        elif self.split == 'valid':
            self.data = data['valid_imgs']
            self.targets = data['valid_labels']
        elif self.split == 'test':
            self.data = data['test_imgs']
            self.targets = data['test_labels']
        
        # Ensure data is in the correct format
        assert self.data.dtype == np.uint8, f"Expected uint8 images, got {self.data.dtype}"
        assert self.targets.dtype == np.uint8, f"Expected uint8 labels, got {self.targets.dtype}"
        assert len(self.data.shape) == 4, f"Expected 4D image array [N,3,H,W], got shape {self.data.shape}"
        assert self.data.shape[1] == 3, f"Expected 3 channels, got {self.data.shape[1]}"
        
        print(f"Loaded {self.split} split: {len(self.data)} samples")
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is the class index.
        """
        img_array = self.data[index]  # Shape: [3, H, W]
        target = int(self.targets[index])
        
        # Convert from CHW to HWC for PIL Image
        img_array = img_array.transpose(1, 2, 0)  # [H, W, 3]
        
        # Create PIL Image
        img = Image.fromarray(img_array, mode='RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self) -> int:
        return len(self.data)
    
    @property
    def class_names(self):
        """Return class names for the colorectal histology dataset."""
        return [
            'tumour epithelium',
            'simple stroma', 
            'complex stroma',
            'immune cell conglomerates',
            'debris and mucus',
            'mucosal glands',
            'adipose tissue',
            'background'
        ]
    
    def extra_repr(self) -> str:
        return f"Split: {self.split}"

class StrainedColorectalHistology(LightningDataModule):
    """
    Lightning DataModule for stain-normalized colorectal histology dataset.
    Uses the StrainedColorectalHistologyDataset which loads from a pre-processed .npz file.
    """
    
    @property
    def num_classes(self):
        return 8

    def __init__(self, 
                 data_dir: str = "./data",
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
        
        self.train_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.get_default_dtype(), scale=True)
        ]

        self.valid_transforms = [
            transforms.ToImage(),
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

        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.output_shape = [batch_size, 3, 150, 150]
        self.num_workers = num_workers

    def prepare_data(self):
        # Check if the stain-normalized dataset exists
        npz_file = os.path.join(self.data_dir, StrainedColorectalHistologyDataset._FILE_NAME)
        if not os.path.exists(npz_file):
            logger.warning(f"Stain-normalized dataset not found at {npz_file}. "
                          "Please run the preprocessing notebook to create it.")

    def setup(self, stage: str):
        self.train_ds = StrainedColorectalHistologyDataset(
            root=self.data_dir,
            split='train',
            transform=self.train_transforms
        )
        self.valid_ds = StrainedColorectalHistologyDataset(
            root=self.data_dir,
            split='valid',
            transform=self.valid_transforms
        )
        self.test_ds = StrainedColorectalHistologyDataset(
            root=self.data_dir,
            split='test',
            transform=self.valid_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, 
            batch_size=self.test_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.test_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

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
