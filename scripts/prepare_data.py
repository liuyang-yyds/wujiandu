"""
Data preparation script for WHU-OPT-SAR dataset
Crops large images into patches and splits into train/val/test sets
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from PIL import Image
import random
from typing import Tuple, List


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare WHU-OPT-SAR dataset')
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to raw WHU-OPT-SAR dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of cropped patches')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between patches')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def crop_patches(
    image_path: str,
    patch_size: int,
    overlap: int
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Crop an image into overlapping patches
    
    Args:
        image_path: Path to image file
        patch_size: Size of each patch
        overlap: Overlap between patches
        
    Returns:
        List of (patch, row_idx, col_idx) tuples
    """
    patches = []
    stride = patch_size - overlap
    
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        
        for row in range(0, height - patch_size + 1, stride):
            for col in range(0, width - patch_size + 1, stride):
                window = Window(col, row, patch_size, patch_size)
                patch = src.read(window=window)
                patches.append((patch, row, col))
    
    return patches


def save_patch(
    patch: np.ndarray,
    output_path: str,
    is_label: bool = False
):
    """
    Save a patch as image file
    
    Args:
        patch: Patch array (C, H, W) or (H, W)
        output_path: Output file path
        is_label: Whether this is a label image
    """
    if patch.ndim == 3:
        # (C, H, W) -> (H, W, C)
        patch = np.transpose(patch, (1, 2, 0))
        
        if patch.shape[2] == 1:
            patch = patch.squeeze(-1)
    
    if is_label:
        # Save as 8-bit label
        Image.fromarray(patch.astype(np.uint8)).save(output_path)
    else:
        # Normalize and save
        if patch.max() > 1:
            patch = patch.astype(np.float32)
            patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        patch = (patch * 255).astype(np.uint8)
        
        if patch.ndim == 2:
            Image.fromarray(patch, mode='L').save(output_path)
        else:
            Image.fromarray(patch).save(output_path)


def process_dataset(
    data_root: str,
    output_dir: str,
    patch_size: int,
    overlap: int,
    train_ratio: float,
    val_ratio: float,
    seed: int
):
    """
    Process the WHU-OPT-SAR dataset
    
    Args:
        data_root: Path to raw dataset
        output_dir: Output directory
        patch_size: Patch size
        overlap: Overlap between patches
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for data_type in ['optical', 'sar', 'labels']:
            (output_dir / data_type / split).mkdir(parents=True, exist_ok=True)
    
    # Find all optical images
    optical_dir = data_root / 'optical'
    sar_dir = data_root / 'sar'
    label_dir = data_root / 'label'
    
    optical_files = sorted(list(optical_dir.glob('*.tif')))
    
    print(f'Found {len(optical_files)} optical images')
    
    all_patches = []
    
    # Process each image
    for opt_file in tqdm(optical_files, desc='Processing images'):
        base_name = opt_file.stem
        
        # Find corresponding SAR and label files
        sar_file = sar_dir / f'{base_name}.tif'
        label_file = label_dir / f'{base_name}.tif'
        
        if not sar_file.exists():
            print(f'Warning: SAR file not found for {base_name}')
            continue
        
        # Crop patches
        opt_patches = crop_patches(str(opt_file), patch_size, overlap)
        sar_patches = crop_patches(str(sar_file), patch_size, overlap)
        
        label_patches = None
        if label_file.exists():
            label_patches = crop_patches(str(label_file), patch_size, overlap)
        
        # Collect patches
        for i, (opt_patch, row, col) in enumerate(opt_patches):
            patch_id = f'{base_name}_{row}_{col}'
            
            sar_patch = sar_patches[i][0]
            label_patch = label_patches[i][0] if label_patches else None
            
            all_patches.append({
                'id': patch_id,
                'optical': opt_patch,
                'sar': sar_patch,
                'label': label_patch
            })
    
    print(f'Total patches: {len(all_patches)}')
    
    # Shuffle and split
    random.shuffle(all_patches)
    
    n_total = len(all_patches)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_patches = all_patches[:n_train]
    val_patches = all_patches[n_train:n_train + n_val]
    test_patches = all_patches[n_train + n_val:]
    
    print(f'Train: {len(train_patches)}, Val: {len(val_patches)}, Test: {len(test_patches)}')
    
    # Save patches
    for split, patches in [('train', train_patches), ('val', val_patches), ('test', test_patches)]:
        print(f'Saving {split} patches...')
        
        for patch_data in tqdm(patches, desc=f'Saving {split}'):
            patch_id = patch_data['id']
            
            # Save optical
            opt_path = output_dir / 'optical' / split / f'{patch_id}.png'
            save_patch(patch_data['optical'], str(opt_path))
            
            # Save SAR
            sar_path = output_dir / 'sar' / split / f'{patch_id}.png'
            save_patch(patch_data['sar'], str(sar_path))
            
            # Save label if available
            if patch_data['label'] is not None:
                label_path = output_dir / 'labels' / split / f'{patch_id}.png'
                save_patch(patch_data['label'], str(label_path), is_label=True)
    
    print('Done!')


def main():
    args = parse_args()
    
    process_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
