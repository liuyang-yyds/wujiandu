"""
Data augmentation transforms for UMSF-Net
Contains transforms for optical and SAR images
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random
from typing import Tuple, Optional, List


class RandomResizedCropPair:
    """
    Apply random resized crop to paired optical and SAR images
    
    Args:
        size (int): Output size
        scale (tuple): Scale range (min, max)
        ratio (tuple): Aspect ratio range
    """
    
    def __init__(
        self,
        size: int = 256,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.33)
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        
    def __call__(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Tuple:
        # Get random crop parameters
        h, w = optical.shape[:2]
        
        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
        aspect_ratio = np.exp(random.uniform(*log_ratio))
        
        crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
        
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        
        # Apply crop
        optical = optical[y:y+crop_h, x:x+crop_w]
        sar = sar[y:y+crop_h, x:x+crop_w]
        
        # Resize
        optical = np.array(
            Image.fromarray(optical).resize((self.size, self.size), Image.BILINEAR)
        )
        sar = np.array(
            Image.fromarray(sar.squeeze()).resize((self.size, self.size), Image.BILINEAR)
        )
        if sar.ndim == 2:
            sar = sar[..., np.newaxis]
        
        if label is not None:
            label = label[y:y+crop_h, x:x+crop_w]
            label = np.array(
                Image.fromarray(label.astype(np.uint8)).resize(
                    (self.size, self.size), Image.NEAREST
                )
            )
            return optical, sar, label
        
        return optical, sar, None


class RandomHorizontalFlipPair:
    """Random horizontal flip for image pairs"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Tuple:
        if random.random() < self.p:
            optical = np.flip(optical, axis=1).copy()
            sar = np.flip(sar, axis=1).copy()
            if label is not None:
                label = np.flip(label, axis=1).copy()
        
        return optical, sar, label


class RandomVerticalFlipPair:
    """Random vertical flip for image pairs"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Tuple:
        if random.random() < self.p:
            optical = np.flip(optical, axis=0).copy()
            sar = np.flip(sar, axis=0).copy()
            if label is not None:
                label = np.flip(label, axis=0).copy()
        
        return optical, sar, label


class RandomRotation90Pair:
    """Random 90-degree rotation for image pairs"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Tuple:
        if random.random() < self.p:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            optical = np.rot90(optical, k, axes=(0, 1)).copy()
            sar = np.rot90(sar, k, axes=(0, 1)).copy()
            if label is not None:
                label = np.rot90(label, k, axes=(0, 1)).copy()
        
        return optical, sar, label


class ComposePair:
    """Compose multiple transforms for image pairs"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
        
    def __call__(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Tuple:
        for t in self.transforms:
            optical, sar, label = t(optical, sar, label)
        return optical, sar, label


class OpticalAugmentation(nn.Module):
    """
    Augmentation pipeline for optical images
    
    Args:
        size (int): Image size
        mean (tuple): Normalization mean
        std (tuple): Normalization std
        training (bool): Whether in training mode
    """
    
    def __init__(
        self,
        size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        training: bool = True
    ):
        super().__init__()
        
        if training:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
    
    def forward(self, x: np.ndarray) -> torch.Tensor:
        return self.transform(x)


class SARAugmentation(nn.Module):
    """
    Augmentation pipeline for SAR images
    
    Args:
        size (int): Image size
        training (bool): Whether in training mode
    """
    
    def __init__(
        self,
        size: int = 256,
        training: bool = True
    ):
        super().__init__()
        self.size = size
        self.training = training
        
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize SAR image to [0, 1]"""
        x = x.astype(np.float32)
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        return x
    
    def _add_speckle(self, x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Add multiplicative speckle noise"""
        noise = np.random.randn(*x.shape).astype(np.float32) * sigma + 1
        return np.clip(x * noise, 0, 1)
    
    def forward(self, x: np.ndarray) -> torch.Tensor:
        # Ensure 2D
        if x.ndim == 3:
            x = x.squeeze()
        
        # Normalize
        x = self._normalize(x)
        
        if self.training:
            # Random speckle noise
            if random.random() < 0.5:
                x = self._add_speckle(x, sigma=random.uniform(0.05, 0.15))
            
            # Random brightness
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                x = np.clip(x * factor, 0, 1)
        
        # To PIL for transforms
        x = Image.fromarray((x * 255).astype(np.uint8))
        
        if self.training:
            x = T.RandomResizedCrop(self.size, scale=(0.2, 1.0))(x)
            if random.random() < 0.5:
                x = TF.hflip(x)
        else:
            x = T.Resize((self.size, self.size))(x)
        
        # To tensor
        x = T.ToTensor()(x)
        
        return x


class ContrastiveAugmentation:
    """
    Augmentation for contrastive learning
    Returns two augmented views
    
    Args:
        optical_aug: Optical augmentation
        sar_aug: SAR augmentation
    """
    
    def __init__(
        self,
        optical_aug: OpticalAugmentation,
        sar_aug: SARAugmentation
    ):
        self.optical_aug = optical_aug
        self.sar_aug = sar_aug
    
    def __call__(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation
        
        Args:
            optical: Optical tensor (C, H, W)
            sar: SAR tensor (1, H, W)
            
        Returns:
            Augmented optical and SAR tensors
        """
        # Convert to numpy for augmentation
        opt_np = (optical.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        sar_np = sar.squeeze().numpy()
        
        # Augment
        opt_aug = self.optical_aug(opt_np)
        sar_aug = self.sar_aug(sar_np)
        
        return opt_aug, sar_aug


def get_train_transforms(size: int = 256):
    """Get training transforms"""
    joint_transform = ComposePair([
        RandomResizedCropPair(size=size, scale=(0.5, 1.0)),
        RandomHorizontalFlipPair(p=0.5),
        RandomVerticalFlipPair(p=0.5),
        RandomRotation90Pair(p=0.5)
    ])
    
    optical_transform = OpticalAugmentation(size=size, training=True)
    sar_transform = SARAugmentation(size=size, training=True)
    
    return joint_transform, optical_transform, sar_transform


def get_val_transforms(size: int = 256):
    """Get validation transforms"""
    optical_transform = OpticalAugmentation(size=size, training=False)
    sar_transform = SARAugmentation(size=size, training=False)
    
    return None, optical_transform, sar_transform
