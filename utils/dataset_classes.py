import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class MelanomaPairedDataset(Dataset):
    """
    Dataset class paired dermoscope and clinical images.
    """
    def __init__(self, df, image_dir, crop_factor=0.5):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.crop_factor = crop_factor  # for clinical images

        # Resize and convert to tensor
        self.final_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def center_crop(self, img, crop_size):
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return img.crop((left, top, left + crop_size, top + crop_size))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        dermo_path = os.path.join(self.image_dir, row["dermo_path"])
        clinical_path = os.path.join(self.image_dir, row["clinical_path"])

        # Open both images
        dermo_img = Image.open(dermo_path).convert("RGB")
        clinical_img = Image.open(clinical_path).convert("RGB")

        # ---- DERMOSCOPE: square crop to min dimension ----
        d_w, d_h = dermo_img.size
        dermo_crop_size = min(d_w, d_h)
        dermo_img = self.center_crop(dermo_img, dermo_crop_size)
        dermo_img = self.final_transform(dermo_img)

        # ---- CLINICAL: crop by factor (e.g., 0.5) ----
        c_w, c_h = clinical_img.size
        clinical_crop_size = int(min(c_w, c_h) * self.crop_factor)
        clinical_img = self.center_crop(clinical_img, clinical_crop_size)
        clinical_img = self.final_transform(clinical_img)

        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "dermoscope": dermo_img,
            "clinical": clinical_img,
            "label": label,
        }

    def __len__(self):
        return len(self.df)


class MelanomaClinicalDataset(Dataset):
    """
    Dataset class for clinical images only.
    """
    def __init__(self, df, image_dir, crop_factor=0.5, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.crop_factor = crop_factor
        self.transform = transform

    def center_crop(self, img, crop_size):
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return img.crop((left, top, left + crop_size, top + crop_size))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        clinical_path = os.path.join(self.image_dir, row["clinical_path"])
        clinical_img = Image.open(clinical_path).convert("RGB")

        # Crop by factor
        c_w, c_h = clinical_img.size
        clinical_crop_size = int(min(c_w, c_h) * self.crop_factor)
        clinical_img = self.center_crop(clinical_img, clinical_crop_size)

        # Apply transform
        if self.transform:
            clinical_img = self.transform(clinical_img)

        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "clinical": clinical_img,
            "label": label,
        }

    def __len__(self):
        return len(self.df)


class ClinicalSquareCropDataset(Dataset):
    """
    Dataset class for clinical images.
    Crops images to centered square (no distortion),
    supports optional transforms.
    Compatible with DataFrame input like MelanomaClinicalDataset.
    """
    def __init__(self, df, image_dir, crop_factor=1.0, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame with 'clinical_path' and 'label' columns
            image_dir (str): root directory prepended to clinical_path
            crop_factor (float): fraction of square side to crop (0 < crop_factor <= 1)
            transform (callable, optional): torchvision transforms to apply
        """
        assert 0 < crop_factor <= 1, "crop_factor must be between 0 and 1"
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.crop_factor = crop_factor
        self.transform = transform

    def center_square_crop(self, img):
        w, h = img.size
        side_len = int(min(w, h) * self.crop_factor)
        left = (w - side_len) // 2
        top = (h - side_len) // 2
        return img.crop((left, top, left + side_len, top + side_len))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["clinical_path"])
        img = Image.open(img_path).convert("RGB")

        # Crop to centered square fraction
        img = self.center_square_crop(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "clinical": img,
            "label": label,
        }

    def __len__(self):
        return len(self.df)
