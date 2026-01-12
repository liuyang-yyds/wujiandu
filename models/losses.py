"""
Loss functions for UMSF-Net
Contains contrastive, clustering, and consistency losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss
    
    Maximizes agreement between positive pairs while minimizing
    agreement with negative samples.
    
    Args:
        temperature (float): Temperature for scaling. Default: 0.07
    """
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Contrastive logits, shape (B, 1 + queue_size)
            labels: Ground truth labels (all zeros for positives)
            
        Returns:
            loss: InfoNCE loss value
        """
        return self.criterion(logits, labels)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR style)
    
    Computes contrastive loss between two views of the same batch.
    
    Args:
        temperature (float): Temperature for scaling. Default: 0.5
    """
    
    def __init__(self, temperature: float = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z1: Features from view 1, shape (B, dim)
            z2: Features from view 2, shape (B, dim)
            
        Returns:
            loss: NT-Xent loss value
        """
        batch_size = z1.size(0)
        
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # (2B, dim)
        
        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z.device)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True
        
        # Compute loss
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        loss = F.cross_entropy(sim, labels)
        
        return loss


class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-modal contrastive loss for optical-SAR alignment
    
    Treats paired optical-SAR images as positive samples and
    non-paired images as negative samples.
    
    Args:
        temperature (float): Temperature for scaling. Default: 0.07
    """
    
    def __init__(self, temperature: float = 0.07):
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(
        self,
        z_optical: torch.Tensor,
        z_sar: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_optical: Optical features, shape (B, dim)
            z_sar: SAR features, shape (B, dim)
            
        Returns:
            loss: Cross-modal contrastive loss
        """
        batch_size = z_optical.size(0)
        
        # Normalize features
        z_optical = F.normalize(z_optical, dim=1)
        z_sar = F.normalize(z_sar, dim=1)
        
        # Cross-modal similarity
        sim = torch.mm(z_optical, z_sar.t()) / self.temperature  # (B, B)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z_optical.device)
        
        # Symmetric loss
        loss_opt2sar = F.cross_entropy(sim, labels)
        loss_sar2opt = F.cross_entropy(sim.t(), labels)
        
        loss = (loss_opt2sar + loss_sar2opt) / 2
        
        return loss


class ClusteringLoss(nn.Module):
    """
    Clustering loss combining entropy minimization and uniformity
    
    Encourages confident and balanced cluster assignments.
    
    Args:
        entropy_weight (float): Weight for entropy minimization. Default: 1.0
        uniformity_weight (float): Weight for uniformity constraint. Default: 1.0
        eps (float): Small constant for numerical stability. Default: 1e-8
    """
    
    def __init__(
        self,
        entropy_weight: float = 1.0,
        uniformity_weight: float = 1.0,
        eps: float = 1e-8
    ):
        super(ClusteringLoss, self).__init__()
        self.entropy_weight = entropy_weight
        self.uniformity_weight = uniformity_weight
        self.eps = eps
        
    def forward(self, p: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            p: Cluster assignment probabilities, shape (B, K)
            
        Returns:
            loss: Total clustering loss
            metrics: Dictionary containing individual loss terms
        """
        batch_size, num_clusters = p.size()
        
        # Entropy minimization: encourage confident predictions
        # H(p) = -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p + self.eps), dim=1)
        entropy_loss = entropy.mean()
        
        # Uniformity: encourage balanced cluster usage
        # Maximize entropy of mean assignment (minimize negative entropy)
        p_mean = p.mean(dim=0)  # (K,)
        target_entropy = torch.log(torch.tensor(num_clusters, dtype=torch.float, device=p.device))
        actual_entropy = -torch.sum(p_mean * torch.log(p_mean + self.eps))
        uniformity_loss = target_entropy - actual_entropy
        
        # Total loss
        loss = self.entropy_weight * entropy_loss + self.uniformity_weight * uniformity_loss
        
        metrics = {
            'entropy': entropy_loss.item(),
            'uniformity': uniformity_loss.item(),
            'p_mean': p_mean.detach()
        }
        
        return loss, metrics


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for multi-view predictions
    
    Encourages consistent cluster assignments across different
    augmented views of the same sample.
    
    Args:
        loss_type (str): Type of consistency loss ('kl', 'mse', 'ce'). Default: 'kl'
    """
    
    def __init__(self, loss_type: str = 'kl'):
        super(ConsistencyLoss, self).__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            p1: Predictions from view 1, shape (B, K)
            p2: Predictions from view 2, shape (B, K)
            
        Returns:
            loss: Consistency loss value
        """
        if self.loss_type == 'kl':
            # Symmetric KL divergence
            loss = F.kl_div(
                torch.log(p1 + 1e-8),
                p2,
                reduction='batchmean'
            ) + F.kl_div(
                torch.log(p2 + 1e-8),
                p1,
                reduction='batchmean'
            )
            loss = loss / 2
            
        elif self.loss_type == 'mse':
            loss = F.mse_loss(p1, p2)
            
        elif self.loss_type == 'ce':
            # Cross entropy with soft targets
            loss = -torch.sum(p2.detach() * torch.log(p1 + 1e-8), dim=1).mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class UMSFLoss(nn.Module):
    """
    Combined loss for UMSF-Net
    
    Combines intra-modal contrastive, cross-modal contrastive,
    clustering, and consistency losses.
    
    Args:
        temperature (float): Temperature for contrastive losses. Default: 0.07
        weights (dict): Loss weights. Default: None
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        weights: Optional[dict] = None
    ):
        super(UMSFLoss, self).__init__()
        
        # Default weights
        if weights is None:
            weights = {
                'intra': 1.0,
                'cross': 1.0,
                'cluster': 0.5,
                'consistency': 0.5
            }
        self.weights = weights
        
        # Loss components
        self.intra_loss = NTXentLoss(temperature)
        self.cross_loss = CrossModalContrastiveLoss(temperature)
        self.cluster_loss = ClusteringLoss()
        self.consistency_loss = ConsistencyLoss()
        
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z_optical: torch.Tensor,
        z_sar: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        use_clustering: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z1, z2: Projected features from two augmented views
            z_optical, z_sar: Projected features from optical and SAR encoders
            p1, p2: Cluster predictions from two views
            use_clustering: Whether to include clustering loss
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss values
        """
        # Intra-modal contrastive loss
        loss_intra = self.intra_loss(z1, z2)
        
        # Cross-modal contrastive loss
        loss_cross = self.cross_loss(z_optical, z_sar)
        
        # Initialize loss dict
        loss_dict = {
            'intra': loss_intra.item(),
            'cross': loss_cross.item()
        }
        
        # Total loss
        total_loss = (
            self.weights['intra'] * loss_intra +
            self.weights['cross'] * loss_cross
        )
        
        if use_clustering:
            # Clustering loss
            loss_cluster, cluster_metrics = self.cluster_loss(p1)
            
            # Consistency loss
            loss_consist = self.consistency_loss(p1, p2)
            
            total_loss += (
                self.weights['cluster'] * loss_cluster +
                self.weights['consistency'] * loss_consist
            )
            
            loss_dict.update({
                'cluster': loss_cluster.item(),
                'consistency': loss_consist.item(),
                'entropy': cluster_metrics['entropy'],
                'uniformity': cluster_metrics['uniformity']
            })
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
