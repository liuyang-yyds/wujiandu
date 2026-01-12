"""
WHU-OPT-SAR Dataset
Dataset loader for optical-SAR paired remote sensing images
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
from typing import Optional, Tuple, List, Callable, Dict
import random


class WHUOptSARDataset(Dataset):
    """
    WHU-OPT-SAR Dataset for optical-SAR image pairs
    
    Args:
        data_root (str): Root directory of the dataset
        split (str): Data split ('train', 'val', 'test')
        optical_transform (callable): Transform for optical images
        sar_transform (callable): Transform for SAR images
        joint_transform (callable): Transform applied to both images jointly
        return_path (bool): Whether to return file paths
    """
    
    # Land cover class names
    CLASSES = [
        'farmland',   # 0
        'city',       # 1
        'village',    # 2
        'water',      # 3
        'forest',     # 4
        'road',       # 5
        'others'      # 6
    ]
    
    # Class colors for visualization (RGB)
    CLASS_COLORS = [
        (0, 128, 0),     # farmland - green
        (255, 0, 0),     # city - red
        (255, 165, 0),   # village - orange
        (0, 0, 255),     # water - blue
        (0, 100, 0),     # forest - dark green
        (255, 255, 255), # road - white
        (128, 128, 128)  # others - gray
    ]
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        optical_transform: Optional[Callable] = None,
        sar_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.optical_transform = optical_transform
        self.sar_transform = sar_transform
        self.joint_transform = joint_transform
        self.return_path = return_path
        
        # Setup paths
        self.optical_dir = os.path.join(data_root, 'optical', split)
        self.sar_dir = os.path.join(data_root, 'sar', split)
        self.label_dir = os.path.join(data_root, 'labels', split)
        
        # Load file list
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, str, str]]:
        """Load list of (optical_path, sar_path, label_path) tuples"""
        samples = []
        
        # Get all optical images
        optical_files = sorted([
            f for f in os.listdir(self.optical_dir)
            if f.endswith(('.tif', '.png', '.jpg'))
        ])
        
        for opt_file in optical_files:
            # Construct corresponding SAR and label paths
            base_name = os.path.splitext(opt_file)[0]
            
            opt_path = os.path.join(self.optical_dir, opt_file)
            
            # Try different extensions for SAR
            sar_path = None
            for ext in ['.tif', '.png', '.jpg']:
                candidate = os.path.join(self.sar_dir, base_name + ext)
                if os.path.exists(candidate):
                    sar_path = candidate
                    break
            
            # Try different extensions for label
            label_path = None
            for ext in ['.tif', '.png']:
                candidate = os.path.join(self.label_dir, base_name + ext)
                if os.path.exists(candidate):
                    label_path = candidate
                    break
            
            if sar_path is not None:
                samples.append((opt_path, sar_path, label_path))
        
        return samples
    
    def _load_image(self, path: str, is_sar: bool = False) -> np.ndarray:
        """Load image from path"""
        if path.endswith('.tif'):
            with rasterio.open(path) as src:
                img = src.read()
                # Convert from (C, H, W) to (H, W, C)
                if img.ndim == 3:
                    img = np.transpose(img, (1, 2, 0))
                return img
        else:
            img = Image.open(path)
            return np.array(img)
    
    def _load_label(self, path: str) -> np.ndarray:
        """Load label mask"""
        if path is None:
            return None
            
        if path.endswith('.tif'):
            with rasterio.open(path) as src:
                label = src.read(1)
        else:
            label = np.array(Image.open(path))
        
        return label.astype(np.int64)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with keys:
                - optical: Optical image tensor (3, H, W)
                - sar: SAR image tensor (1, H, W)
                - label: Label tensor (H, W) or None
                - path: File paths (optional)
        """
        opt_path, sar_path, label_path = self.samples[idx]
        
        # Load images
        optical = self._load_image(opt_path, is_sar=False)
        sar = self._load_image(sar_path, is_sar=True)
        label = self._load_label(label_path)
        
        # Ensure correct dimensions
        if optical.ndim == 2:
            optical = np.stack([optical] * 3, axis=-1)
        elif optical.shape[-1] > 3:
            optical = optical[..., :3]
        
        if sar.ndim == 3:
            sar = np.mean(sar, axis=-1, keepdims=True)
        elif sar.ndim == 2:
            sar = sar[..., np.newaxis]
        
        # Joint transform (e.g., random crop, flip)
        if self.joint_transform is not None:
            optical, sar, label = self.joint_transform(optical, sar, label)
        
        # Individual transforms
        if self.optical_transform is not None:
            optical = self.optical_transform(optical)
        else:
            optical = torch.from_numpy(optical.transpose(2, 0, 1)).float() / 255.0
        
        if self.sar_transform is not None:
            sar = self.sar_transform(sar)
        else:
            sar = torch.from_numpy(sar.transpose(2, 0, 1)).float()
            # Normalize SAR to [0, 1]
            sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-8)
        
        # Prepare output
        output = {
            'optical': optical,
            'sar': sar,
        }
        
        if label is not None:
            output['label'] = torch.from_numpy(label).long()
        
        if self.return_path:
            output['path'] = (opt_path, sar_path)
        
        return output


class ContrastiveDataset(Dataset):
    """
    Dataset wrapper for contrastive learning
    Returns two augmented views of each sample
    
    Args:
        base_dataset: Base WHUOptSARDataset
        augmentation: Augmentation function that returns two views
    """
    
    def __init__(
        self,
        base_dataset: WHUOptSARDataset,
        augmentation: Callable
    ):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.base_dataset[idx]
        
        # Apply augmentation to get two views
        opt1, sar1 = self.augmentation(sample['optical'], sample['sar'])
        opt2, sar2 = self.augmentation(sample['optical'], sample['sar'])
        
        output = {
            'optical_1': opt1,
            'optical_2': opt2,
            'sar_1': sar1,
            'sar_2': sar2,
        }
        
        if 'label' in sample:
            output['label'] = sample['label']
        
        return output


def create_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    optical_transform: Optional[Callable] = None,
    sar_transform: Optional[Callable] = None,
    joint_transform: Optional[Callable] = None,
    shuffle: bool = True
) -> DataLoader:
    """
    Create dataloader for WHU-OPT-SAR dataset
    
    Args:
        data_root: Root directory of dataset
        split: Data split
        batch_size: Batch size
        num_workers: Number of data loading workers
        optical_transform: Transform for optical images
        sar_transform: Transform for SAR images
        joint_transform: Joint transform
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader instance
    """
    dataset = WHUOptSARDataset(
        data_root=data_root,
        split=split,
        optical_transform=optical_transform,
        sar_transform=sar_transform,
        joint_transform=joint_transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return loader
