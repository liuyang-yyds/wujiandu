"""
Training script for UMSF-Net
Unsupervised Multi-Source Fusion Network for Land Cover Classification
"""

import os
import argparse
import yaml
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from models import build_moco_model, UMSFLoss
from datasets import (
    WHUOptSARDataset,
    ContrastiveDataset,
    get_train_transforms,
    get_val_transforms,
    ContrastiveAugmentation,
    OpticalAugmentation,
    SARAugmentation
)
from utils.metrics import ClusteringMetrics
from utils.logger import Logger
from utils.misc import AverageMeter, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train UMSF-Net')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_optimizer(model, config):
    """Build optimizer from config"""
    opt_config = config['training']['optimizer']
    
    if opt_config['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 1e-4)
        )
    elif opt_config['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    return optimizer


def build_scheduler(optimizer, config, num_iterations):
    """Build learning rate scheduler"""
    sched_config = config['training']['scheduler']
    
    if sched_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_iterations,
            eta_min=sched_config.get('min_lr', 1e-6)
        )
    elif sched_config['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 30),
            gamma=sched_config.get('gamma', 0.1)
        )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    config,
    logger,
    writer
):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    loss_intra = AverageMeter()
    loss_cross = AverageMeter()
    loss_cluster = AverageMeter()
    
    clustering_start = config['training'].get('clustering_start_epoch', 50)
    use_clustering = epoch >= clustering_start
    
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        opt_1 = batch['optical_1'].cuda(non_blocking=True)
        opt_2 = batch['optical_2'].cuda(non_blocking=True)
        sar_1 = batch['sar_1'].cuda(non_blocking=True)
        sar_2 = batch['sar_2'].cuda(non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['training'].get('mixed_precision', True)):
            # Query and key forward (for MoCo contrastive)
            logits, labels, p1 = model(opt_1, sar_1, opt_2, sar_2)
            
            # Get projected features for view 1
            z1, _, features = model.encoder_q(opt_1, sar_1, return_features=True)
            
            # Get projected features for view 2
            with torch.no_grad():
                z2, p2, _ = model.encoder_q(opt_2, sar_2, return_features=True)
            
            # Compute loss (z1, z2 are projected features with shape (B, dim))
            loss_info = criterion(
                z1,
                z2,
                features['optical'],
                features['sar'],
                p1,
                p2,
                use_clustering=use_clustering
            )
            
            # InfoNCE loss from MoCo
            moco_loss = nn.CrossEntropyLoss()(logits, labels)
            loss = moco_loss + loss_info[0]
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update meters
        batch_size = opt_1.size(0)
        losses.update(loss.item(), batch_size)
        loss_intra.update(loss_info[1].get('intra', 0), batch_size)
        loss_cross.update(loss_info[1].get('cross', 0), batch_size)
        if use_clustering:
            loss_cluster.update(loss_info[1].get('cluster', 0), batch_size)
        
        # Log
        if batch_idx % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_batches}] '
                f'Loss: {losses.avg:.4f} '
                f'Intra: {loss_intra.avg:.4f} '
                f'Cross: {loss_cross.avg:.4f} '
                f'Cluster: {loss_cluster.avg:.4f} '
                f'LR: {lr:.6f}'
            )
    
    # TensorBoard logging
    global_step = epoch * num_batches
    writer.add_scalar('train/loss', losses.avg, global_step)
    writer.add_scalar('train/loss_intra', loss_intra.avg, global_step)
    writer.add_scalar('train/loss_cross', loss_cross.avg, global_step)
    writer.add_scalar('train/loss_cluster', loss_cluster.avg, global_step)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    return losses.avg


@torch.no_grad()
def evaluate(model, val_loader, epoch, logger, writer):
    """Evaluate model"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_features = []
    
    for batch in val_loader:
        optical = batch['optical'].cuda(non_blocking=True)
        sar = batch['sar'].cuda(non_blocking=True)
        
        # Forward
        z, p, features = model.encoder_q(optical, sar, return_features=True)
        
        all_predictions.append(p.argmax(dim=1).cpu())
        all_features.append(features['fused'].cpu())
        
        if 'label' in batch:
            all_labels.append(batch['label'])
    
    predictions = torch.cat(all_predictions, dim=0).numpy()
    features = torch.cat(all_features, dim=0).numpy()
    
    if len(all_labels) > 0:
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Compute metrics
        metrics = ClusteringMetrics()
        acc = metrics.clustering_accuracy(labels.flatten(), predictions.flatten())
        nmi = metrics.nmi(labels.flatten(), predictions.flatten())
        ari = metrics.ari(labels.flatten(), predictions.flatten())
        
        logger.info(
            f'Eval Epoch [{epoch}] '
            f'ACC: {acc:.4f} '
            f'NMI: {nmi:.4f} '
            f'ARI: {ari:.4f}'
        )
        
        # TensorBoard
        writer.add_scalar('eval/acc', acc, epoch)
        writer.add_scalar('eval/nmi', nmi, epoch)
        writer.add_scalar('eval/ari', ari, epoch)
        
        return acc
    
    return 0.0


def main():
    args = parse_args()
    
    # Setup distributed training
    if args.distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup logger and tensorboard
    logger = Logger(output_dir / 'train.log')
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    logger.info(f'Config: {config}')
    logger.info(f'Output directory: {output_dir}')
    
    # Build model
    model = build_moco_model(config['model'])
    model = model.cuda()
    
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Build criterion
    criterion = UMSFLoss(
        temperature=config['training']['contrastive']['temperature'],
        weights=config['training'].get('loss_weights', None)
    )
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    
    # Build dataloaders
    joint_train, opt_train, sar_train = get_train_transforms(
        size=config['data'].get('image_size', 256)
    )
    _, opt_val, sar_val = get_val_transforms(
        size=config['data'].get('image_size', 256)
    )
    
    train_dataset = WHUOptSARDataset(
        data_root=args.data_root,
        split='train',
        joint_transform=joint_train
    )
    
    # Wrap with contrastive augmentation
    contrastive_aug = ContrastiveAugmentation(
        optical_aug=OpticalAugmentation(training=True),
        sar_aug=SARAugmentation(training=True)
    )
    train_dataset = ContrastiveDataset(train_dataset, contrastive_aug)
    
    val_dataset = WHUOptSARDataset(
        data_root=args.data_root,
        split='val',
        optical_transform=opt_val,
        sar_transform=sar_val
    )
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Build scheduler
    num_iterations = len(train_loader) * config['training']['epochs']
    scheduler = build_scheduler(optimizer, config, num_iterations)
    
    # Mixed precision
    scaler = GradScaler(enabled=config['training'].get('mixed_precision', True))
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        logger.info(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, epoch, config, logger, writer
        )
        
        # Evaluate
        if epoch % config.get('evaluation', {}).get('eval_interval', 5) == 0:
            acc = evaluate(model, val_loader, epoch, logger, writer)
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(
                    output_dir / 'best_model.pth',
                    model, optimizer, scheduler, epoch, best_acc
                )
                logger.info(f'New best accuracy: {best_acc:.4f}')
        
        # Save checkpoint
        if epoch % config.get('evaluation', {}).get('save_interval', 10) == 0:
            save_checkpoint(
                output_dir / f'checkpoint_epoch_{epoch}.pth',
                model, optimizer, scheduler, epoch, best_acc
            )
    
    # Save final model
    save_checkpoint(
        output_dir / 'final_model.pth',
        model, optimizer, scheduler, config['training']['epochs'] - 1, best_acc
    )
    
    logger.info(f'Training completed. Best accuracy: {best_acc:.4f}')
    writer.close()


if __name__ == '__main__':
    main()