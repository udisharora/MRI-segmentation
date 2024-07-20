import os
import numpy as np
import nibabel as nib
import cv2
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, image_size=128, transform=None):
        self.data_dir = data_dir
        self.IDs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        ID = self.IDs[idx]
        path = os.path.join(self.data_dir, ID)
        t1c_path = os.path.join(path, f'{ID}-t1c.nii.gz')
        t1_path = os.path.join(path, f'{ID}-t1n.nii.gz')
        t2_path = os.path.join(path, f'{ID}-t2f.nii.gz')
        flair_path = os.path.join(path, f'{ID}-t2w.nii.gz')
        seg_path = os.path.join(path, f'{ID}-seg.nii.gz')

        # Load NIFTI files
        t1c_data = nib.load(t1c_path).get_fdata()
        t1_data = nib.load(t1_path).get_fdata()
        t2_data = nib.load(t2_path).get_fdata()
        flair_data = nib.load(flair_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()

        # Assuming all modalities have the same shape, we choose a middle slice
        z_index = seg_data.shape[-1] // 2
        t1c_slice = t1c_data[:, :, z_index]
        t1_slice = t1_data[:, :, z_index]
        t2_slice = t2_data[:, :, z_index]
        flair_slice = flair_data[:, :, z_index]
        seg_slice = seg_data[:, :, z_index]

        # Resize images to image_size
        t1c_slice = cv2.resize(t1c_slice, (self.image_size, self.image_size))
        t1_slice = cv2.resize(t1_slice, (self.image_size, self.image_size))
        t2_slice = cv2.resize(t2_slice, (self.image_size, self.image_size))
        flair_slice = cv2.resize(flair_slice, (self.image_size, self.image_size))
        seg_slice = cv2.resize(seg_slice, (self.image_size, self.image_size))

        # Stack modalities to create multi-channel input
        X = np.stack([t1c_slice, t1_slice, t2_slice, flair_slice], axis=-1)

        # Normalize the input data
        X = X / np.max(X)

        # Convert segmentation mask to tensor of class indices
        y = torch.tensor(seg_slice, dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            X = self.transform(X)

        return X, y

