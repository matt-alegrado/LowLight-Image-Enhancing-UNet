import random

import os
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from lightning.pytorch import LightningDataModule
import torch
import re
import math

class SIDDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None, transform_debug=None, iso_max=25600, max_exposure_time=0.1):
        """
        Args:
            root_dir (str): base path to dataset folder (containing 'Sony' subfolders)
            list_file (str): path to train/val/test split file, e.g. 'Sony_train_list.txt'
            transform (callable, optional): torchvision transforms for PIL Images
            transform_debug (callable, optional): debug transforms (if any)
            flip (bool): whether to randomly flip the image pairs
        """
        self.root_dir = root_dir
        self.transform = transform
        self.transform_debug = transform_debug
        self.entries = []  # will hold tuples (short_path, long_path, iso, exposure_time)
        self.iso_max = iso_max
        self.max_exposure_time = max_exposure_time

        # Regex to capture “_X.XXs” or “_Xs” in a filename:
        #   - group 1: one or more digits, possibly with a decimal point, followed by 's'
        #   e.g. “_0.04s” or “_10s”
        exposure_regex = re.compile(r"_(\d+(?:\.\d+)?)s", re.IGNORECASE)

        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                short_rel, long_rel = parts[0], parts[1]
                iso_token = parts[2]  # e.g. 'ISO200'
                # parts[3] used to be aperture (e.g. 'F13'), but now we will ignore that

                # 1) Convert “.ARW” (or “.arw”) to “.png” in the relative paths
                short_rel = short_rel.replace('.ARW', '.png').replace('.arw', '.png')
                long_rel = long_rel.replace('.ARW', '.png').replace('.arw', '.png')

                # 2) Ensure we point to “Sony_PNG” instead of “Sony”
                short_rel = short_rel.replace('Sony/', 'Sony_PNG/')
                long_rel = long_rel.replace('Sony/', 'Sony_PNG/')

                # 3) Build full filesystem paths
                short_path = os.path.join(root_dir, short_rel.lstrip('./'))
                long_path = os.path.join(root_dir, long_rel.lstrip('./'))

                # 4) Parse numeric ISO from iso_token (e.g. 'ISO200' → 200)
                try:
                    iso = int(iso_token.replace('ISO', ''))
                except ValueError:
                    raise ValueError(f"Cannot parse ISO from token '{iso_token}' in line: {line}")

                # 5) Parse exposure time out of the short‐exposure filename
                #    We look for something like '_0.04s' or '_10s' in short_rel.
                m = exposure_regex.search(short_rel)
                if not m:
                    raise ValueError(f"Cannot find exposure string in '{short_rel}'")
                exposure_time = float(m.group(1))  # e.g. '0.04' or '10' -> 0.04 or 10.0

                # Now store a tuple of (short_path, long_path, iso, exposure_time)
                self.entries.append((short_path, long_path, iso, exposure_time))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        short_path, long_path, iso, exposure_time = self.entries[idx]

        # ISO noise scales approximately with its sqrt, turn that into an input channel (max ISO: 25600)
        iso_norm = float(iso**.5 / self.iso_max**.5)

        exposure_time_norm = float(exposure_time / self.max_exposure_time)

        # Now assuming .png entries
        img_short = Image.open(short_path).convert('RGB')  # convert to RGB to ensure consistency
        img_long = Image.open(long_path).convert('RGB')

        # apply transforms if any
        if self.transform:
            img_short = self.transform(img_short)
            img_long  = self.transform(img_long)
            if self.transform_debug:
                img_short = self.transform_debug(img_long)

        # metadata = {'iso': iso_norm, 'exposure': exposure_time}
        # Add 4th and 5th channels to img_short, for iso and exposure time data
        H, W = img_short.shape[1], img_short.shape[2]
        exposure_map = torch.full((1, H, W), fill_value=exposure_time_norm, dtype=torch.float32)
        iso_map = torch.full((1, H, W), fill_value=iso_norm, dtype=torch.float32)

        img_short_augmented = torch.cat((img_short, iso_map, exposure_map), dim=0)

        return img_short_augmented, img_long

class SIDDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._subset_indices = list(range(len(self.train_dataset)))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )