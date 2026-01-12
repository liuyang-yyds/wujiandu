"""
Evaluation script for UMSF-Net
"""

import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from datasets import WHUOptSARDataset, get_val_transforms
from utils.metrics import ClusteringMetrics
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate UMSF-Net')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on dataset
    
    Returns:
        predictions: Cluster predictions
        labels: Ground truth labels
        features: Extracted features
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_features = []
    all_probs = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        optical = batch['optical'].to(device)
        sar = batch['sar'].to(device)
        
        # Forward pass
        z, p, features = model(optical, sar, return_features=True)
        
        all_predictions.append(p.argmax(dim=1).cpu())
        all_probs.append(p.cpu())
        all_features.append(features['fused'].cpu())
        
        if 'label' in batch:
            all_labels.append(batch['label'])
    
    predictions = torch.cat(all_predictions, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    features = torch.cat(all_features, dim=0).numpy()
    
    labels = None
    if len(all_labels) > 0:
        labels = torch.cat(all_labels, dim=0).numpy()
    
    return predictions, labels, features, probs


def compute_metrics(predictions, labels, metrics_calculator):
    """Compute clustering metrics"""
    # Flatten if needed
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if labels.ndim > 1:
        labels = labels.flatten()
    
    # Remove invalid labels
    valid_mask = labels >= 0
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    results = {
        'accuracy': metrics_calculator.clustering_accuracy(labels, predictions),
        'nmi': metrics_calculator.nmi(labels, predictions),
        'ari': metrics_calculator.ari(labels, predictions),
        'f1_macro': metrics_calculator.f1_score(labels, predictions, average='macro'),
        'f1_weighted': metrics_calculator.f1_score(labels, predictions, average='weighted'),
    }
    
    # Per-class metrics
    per_class_f1 = metrics_calculator.f1_score(labels, predictions, average=None)
    for i, f1 in enumerate(per_class_f1):
        results[f'f1_class_{i}'] = f1
    
    return results


def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(output_dir / 'evaluate.log')
    logger.info(f'Evaluating on {args.split} split')
    logger.info(f'Checkpoint: {args.checkpoint}')
    
    # Load config
    config = load_config(args.config)
    
    # Build model
    model = build_model(config['model'])
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DDP state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        elif k.startswith('encoder_q.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    logger.info('Model loaded successfully')
    
    # Build dataloader
    _, opt_transform, sar_transform = get_val_transforms(
        size=config['data'].get('image_size', 256)
    )
    
    dataset = WHUOptSARDataset(
        data_root=args.data_root,
        split=args.split,
        optical_transform=opt_transform,
        sar_transform=sar_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f'Dataset size: {len(dataset)}')
    
    # Evaluate
    predictions, labels, features, probs = evaluate(model, dataloader, device)
    
    # Compute metrics
    if labels is not None:
        metrics_calculator = ClusteringMetrics()
        results = compute_metrics(predictions, labels, metrics_calculator)
        
        logger.info('=== Evaluation Results ===')
        for metric, value in results.items():
            logger.info(f'{metric}: {value:.4f}')
        
        # Save results
        np.save(output_dir / 'predictions.npy', predictions)
        np.save(output_dir / 'labels.npy', labels)
        np.save(output_dir / 'features.npy', features)
        np.save(output_dir / 'probs.npy', probs)
        
        with open(output_dir / 'metrics.yaml', 'w') as f:
            yaml.dump(results, f)
        
        logger.info(f'Results saved to {output_dir}')
    else:
        logger.warning('No labels available, skipping metric computation')
        np.save(output_dir / 'predictions.npy', predictions)
        np.save(output_dir / 'features.npy', features)


if __name__ == '__main__':
    main()
