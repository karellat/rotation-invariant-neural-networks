import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from hippy2d.datasets import StrainedColorectalHistologyDataset, StrainedColorectalHistology 

class TestDatasets:
    """Test dataset loading and integrity."""
    def test_strainedhistologyds(self):
        train_ds = StrainedColorectalHistologyDataset(root="hippy2d/data", split="train")
        valid_ds = StrainedColorectalHistologyDataset(root="hippy2d/data", split="valid")
        test_ds = StrainedColorectalHistologyDataset(root="hippy2d/data", split="test")

        for ds in [train_ds, valid_ds, test_ds]: 
            ds_samples = len(ds)
            for i in tqdm(range(len(ds)), desc="Testing StrainedColorectalHistologyDataset"):
                # Calculate the number of samples
                img, label = ds[i]
                assert isinstance(img, Image.Image), f"Image at index {i} is not a PIL Image"
                img = np.array(img)
                assert isinstance(img, np.ndarray), f"Image at index {i} is not a numpy array"
                assert img.ndim == 3 and img.shape[2] == 3, f"Image at index {i} does not have 3 channels"
                assert img.min() >= 0 and img.max() <= 255, f"Image at index {i} has pixel values out of bounds [0, 255]"
                assert isinstance(label, int), f"Label at index {i} is not an integer"
                assert 0 <= label < 8, f"Label at index {i} is out of bounds [0, 7]"
            print(f"Dataset split '{ds.split}' passed all tests with {ds_samples} samples.")
    
    def test_strainedhistology(self):
        # Test the StrainedColorectalHistology LightningDataModule
        dm = StrainedColorectalHistology(data_dir="hippy2d/data", batch_size=64, num_workers=1)
        dm.setup("fit")
        dm.prepare_data()
        # go through all loaders
        for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            for batch in tqdm(loader, desc="Testing StrainedColorectalHistology DataLoader"):
                imgs, labels = batch
                assert isinstance(imgs, torch.Tensor), "Images batch is not a torch Tensor"
                assert imgs.ndim == 4 and imgs.shape[1] == 3, "Images batch does not have shape (B, 3, H, W)"
                assert isinstance(labels, torch.Tensor), "Labels batch is not a torch Tensor"
                assert labels.ndim == 1, "Labels batch is not 1-dimensional"
                assert labels.min() >= 0 and labels.max() < 8, "Labels are out of bounds [0, 7]"
            print("DataLoader passed all tests.")