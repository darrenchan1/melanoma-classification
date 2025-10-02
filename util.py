import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import os
import shutil
from torch.utils.data import Dataset
from torchvision import transforms
import torch


def load_train_and_val_df(image_directory, metadata_csv):
    """
    Load training and validation data from a CSV file and image directory.

    Args:
        image_directory: Path to the directory containing images.
        metadata_csv: Path to the CSV file containing metadata.
    Returns:
        train_df: DataFrame containing training data.
        val_df: DataFrame containing validation data.
    """
    df = pd.read_csv(metadata_csv)

    # Remove control examples
    df = df[df['midas_iscontrol'].str.lower() == 'no']

    # Normalize casing
    df['midas_distance'] = df['midas_distance'].str.lower()

    # Determine modality
    df['modality'] = df['midas_distance'].apply(
        lambda x: 'dermoscope' if x == 'dscope' else ('clinical' if isinstance(x, str) else None)
    )

    # Add existence flag
    df['image_path'] = df['midas_file_name'].apply(lambda x: os.path.join(image_directory, x))
    df['file_exists'] = df['image_path'].apply(os.path.exists)

    # Keep only valid images
    df = df[df['file_exists'] & df['modality'].isin(['dermoscope', 'clinical'])]

    # Only keep patients with at least 1 of each modality
    grouped = df.groupby('midas_record_id')
    valid_patients = [
        pid for pid, g in grouped
        if 'dermoscope' in g['modality'].values and 'clinical' in g['modality'].values
    ]

    # Filter down
    df = df[df['midas_record_id'].isin(valid_patients)]

    # For each patient, sample exactly 1 of each modality
    def sample_patient_rows(group):
        dermo = group[group['modality'] == 'dermoscope'].sample(1)
        clinical = group[group['modality'] == 'clinical'].sample(1)
        return pd.concat([dermo, clinical])

    final_df = df.groupby('midas_record_id').apply(sample_patient_rows).reset_index(drop=True)

    grouped = df.groupby('midas_record_id')
    valid_patients = [
        pid for pid, g in grouped
        if 'dermoscope' in g['modality'].values and 'clinical' in g['modality'].values
    ]

    # Assign labels: 1 = malignant, 0 = benign
    # Only use rows that have path info (non-control)
    final_df = final_df[final_df['midas_iscontrol'].str.lower() == 'no']
    final_df['label'] = final_df['midas_path'].str.lower().str.contains("malig").astype(int)

    # Restructure into per-patient rows
    patients = []
    for pid, g in final_df.groupby('midas_record_id'):
        if len(g) != 2: continue
        dermo_row = g[g['modality'] == 'dermoscope'].iloc[0]
        clin_row = g[g['modality'] == 'clinical'].iloc[0]
        label = dermo_row['label']  # same for both rows
        patients.append({
            'midas_record_id': pid,
            'dermo_path': dermo_row['image_path'],
            'clinical_path': clin_row['image_path'],
            'dermo_midas_distance': dermo_row['midas_distance'],
            'clinical_midas_distance': clin_row['midas_distance'],
            'label': label
        })

    patient_df = pd.DataFrame(patients)

    # Train/Val Split (patient-level)
    train_df, val_df = train_test_split(
        patient_df, test_size=0.2, random_state=42, stratify=patient_df['label']
    )

    return train_df, val_df


def load_train_val_test_df(image_directory, metadata_csv, val_size=0.1, test_size=0.2, random_state=42):
    """
    Load training, validation, and test data from a CSV file and image directory.

    Args:
        image_directory: Path to the directory containing images.
        metadata_csv: Path to the CSV file containing metadata.
        val_size: Proportion of train_val to use as validation.
        test_size: Proportion of total data to use as test set.
        random_state: Random seed for reproducibility.
    Returns:
        train_df, val_df, test_df: DataFrames for train, validation, and test sets.
    """
    df = pd.read_csv(metadata_csv)

    # Remove control examples
    df = df[df['midas_iscontrol'].str.lower() == 'no']

    # Normalize casing
    df['midas_distance'] = df['midas_distance'].str.lower()

    # Determine modality
    df['modality'] = df['midas_distance'].apply(
        lambda x: 'dermoscope' if x == 'dscope' else ('clinical' if isinstance(x, str) else None)
    )

    # Add image existence flag
    df['image_path'] = df['midas_file_name'].apply(lambda x: os.path.join(image_directory, x))
    df['file_exists'] = df['image_path'].apply(os.path.exists)

    # Keep valid rows only
    df = df[df['file_exists'] & df['modality'].isin(['dermoscope', 'clinical'])]

    # Only keep patients with both dermoscope and clinical images
    grouped = df.groupby('midas_record_id')
    valid_patients = [
        pid for pid, g in grouped
        if 'dermoscope' in g['modality'].values and 'clinical' in g['modality'].values
    ]
    df = df[df['midas_record_id'].isin(valid_patients)]

    # Sample exactly 1 of each modality per patient
    def sample_patient_rows(group):
        dermo = group[group['modality'] == 'dermoscope'].sample(1, random_state=random_state)
        clinical = group[group['modality'] == 'clinical'].sample(1, random_state=random_state)
        return pd.concat([dermo, clinical])

    final_df = df.groupby('midas_record_id').apply(sample_patient_rows).reset_index(drop=True)

    # Assign binary labels
    final_df['label'] = final_df['midas_path'].str.lower().str.contains("malig").astype(int)

    # Create per-patient rows
    patients = []
    for pid, g in final_df.groupby('midas_record_id'):
        if len(g) != 2: continue
        dermo_row = g[g['modality'] == 'dermoscope'].iloc[0]
        clin_row = g[g['modality'] == 'clinical'].iloc[0]
        label = dermo_row['label']
        patients.append({
            'midas_record_id': pid,
            'dermo_path': dermo_row['image_path'],
            'clinical_path': clin_row['image_path'],
            'dermo_midas_distance': dermo_row['midas_distance'],
            'clinical_midas_distance': clin_row['midas_distance'],
            'label': label
        })

    patient_df = pd.DataFrame(patients)

    # First split into train_val and test
    train_val_df, test_df = train_test_split(
        patient_df, test_size=test_size, random_state=random_state, stratify=patient_df['label']
    )

    # Then split train_val into train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['label']
    )

    return train_df, val_df, test_df


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
    

# class MelanomaClinicalDataset(Dataset):
#     """
#     Dataset class for clinical images only.
#     """
#     def __init__(self, df, image_dir, crop_factor=0.5):
#         self.df = df.reset_index(drop=True)
#         self.image_dir = image_dir
#         self.crop_factor = crop_factor  # for cropping the clinical image

#         self.final_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])

#     def center_crop(self, img, crop_size):
#         w, h = img.size
#         left = (w - crop_size) // 2
#         top = (h - crop_size) // 2
#         return img.crop((left, top, left + crop_size, top + crop_size))

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]

#         clinical_path = os.path.join(self.image_dir, row["clinical_path"])
#         clinical_img = Image.open(clinical_path).convert("RGB")

#         # Crop by factor
#         c_w, c_h = clinical_img.size
#         clinical_crop_size = int(min(c_w, c_h) * self.crop_factor)
#         clinical_img = self.center_crop(clinical_img, clinical_crop_size)
#         clinical_img = self.final_transform(clinical_img)

#         label = torch.tensor(row["label"], dtype=torch.long)

#         return {
#             "clinical": clinical_img,
#             "label": label,
#         }

#     def __len__(self):
#         return len(self.df)


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
