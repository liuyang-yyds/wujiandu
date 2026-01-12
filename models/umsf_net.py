"""
UMSF-Net: Unsupervised Multi-Source Fusion Network
Main network architecture for land cover classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import OpticalEncoder, SAREncoder
from .fusion import CrossModalAttentionFusion
from .heads import ProjectionHead, ClusteringHead


class UMSFNet(nn.Module):
    """
    Unsupervised Multi-Source Fusion Network for Land Cover Classification
    
    This network fuses optical and SAR remote sensing images through:
    1. Dual-branch feature extraction (optical & SAR encoders)
    2. Cross-modal attention fusion
    3. Contrastive learning with projection head
    4. Unsupervised clustering head
    
    Args:
        backbone (str): Backbone network type. Default: 'resnet50'
        pretrained (bool): Whether to use ImageNet pretrained weights. Default: True
        feature_dim (int): Feature dimension from encoders. Default: 2048
        projection_dim (int): Projection head output dimension. Default: 256
        num_classes (int): Number of land cover classes. Default: 7
        num_heads (int): Number of attention heads in fusion module. Default: 8
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 2048,
        projection_dim: int = 256,
        num_classes: int = 7,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(UMSFNet, self).__init__()
        
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        
        # Dual-branch encoders
        self.optical_encoder = OpticalEncoder(
            backbone=backbone,
            pretrained=pretrained,
            out_dim=feature_dim
        )
        
        self.sar_encoder = SAREncoder(
            backbone=backbone,
            pretrained=pretrained,
            out_dim=feature_dim,
            use_despeckling=True
        )
        
        # Cross-modal attention fusion
        self.fusion = CrossModalAttentionFusion(
            in_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            in_dim=feature_dim,
            hidden_dim=feature_dim,
            out_dim=projection_dim
        )
        
        # Clustering head for unsupervised classification
        self.clustering_head = ClusteringHead(
            in_dim=feature_dim,
            hidden_dim=512,
            num_classes=num_classes
        )
        
    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        return_features: bool = False
    ):
        """
        Forward pass of UMSF-Net
        
        Args:
            optical: Optical image tensor, shape (B, 3, H, W)
            sar: SAR image tensor, shape (B, 1, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            If return_features=False:
                z: Projected features for contrastive learning, shape (B, projection_dim)
                p: Cluster assignment probabilities, shape (B, num_classes)
            If return_features=True:
                z, p, features_dict
        """
        # Extract features from both branches
        f_optical = self.optical_encoder(optical)  # (B, feature_dim)
        f_sar = self.sar_encoder(sar)              # (B, feature_dim)
        
        # Cross-modal attention fusion
        f_fused, attention_weights = self.fusion(f_optical, f_sar)  # (B, feature_dim)
        
        # Projection for contrastive learning
        z = self.projection_head(f_fused)  # (B, projection_dim)
        z = F.normalize(z, dim=1)          # L2 normalization
        
        # Clustering prediction
        p = self.clustering_head(f_fused)  # (B, num_classes)
        
        if return_features:
            features = {
                'optical': f_optical,
                'sar': f_sar,
                'fused': f_fused,
                'attention_weights': attention_weights
            }
            return z, p, features
        
        return z, p
    
    def encode(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        modality: str = 'fused'
    ):
        """
        Extract features without projection or clustering
        
        Args:
            optical: Optical image tensor
            sar: SAR image tensor
            modality: Which features to return ('optical', 'sar', 'fused')
            
        Returns:
            features: Feature tensor, shape (B, feature_dim)
        """
        f_optical = self.optical_encoder(optical)
        f_sar = self.sar_encoder(sar)
        
        if modality == 'optical':
            return f_optical
        elif modality == 'sar':
            return f_sar
        else:
            f_fused, _ = self.fusion(f_optical, f_sar)
            return f_fused
    
    def get_cluster_centers(self):
        """Get learned cluster centers from clustering head"""
        return self.clustering_head.get_centers()


class UMSFNetMoCo(nn.Module):
    """
    UMSF-Net with Momentum Contrast (MoCo v3 style)
    
    Maintains a momentum-updated encoder and a queue of negative samples
    for stable contrastive learning.
    
    Args:
        base_model (UMSFNet): Base UMSF-Net model
        momentum (float): Momentum for updating key encoder. Default: 0.999
        queue_size (int): Size of negative sample queue. Default: 65536
        temperature (float): Temperature for contrastive loss. Default: 0.07
    """
    
    def __init__(
        self,
        base_model: UMSFNet,
        momentum: float = 0.999,
        queue_size: int = 65536,
        temperature: float = 0.07
    ):
        super(UMSFNetMoCo, self).__init__()
        
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        
        # Query encoder (trainable)
        self.encoder_q = base_model
        
        # Key encoder (momentum-updated)
        self.encoder_k = UMSFNet(
            backbone='resnet50',
            pretrained=True,
            feature_dim=base_model.feature_dim,
            projection_dim=base_model.projection_dim,
            num_classes=base_model.num_classes
        )
        
        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Create queue for negative samples
        self.register_buffer(
            'queue',
            F.normalize(torch.randn(base_model.projection_dim, queue_size), dim=0)
        )
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(
        self,
        optical_q: torch.Tensor,
        sar_q: torch.Tensor,
        optical_k: torch.Tensor,
        sar_k: torch.Tensor
    ):
        """
        Forward pass with query and key views
        
        Args:
            optical_q, sar_q: Query view (augmented version 1)
            optical_k, sar_k: Key view (augmented version 2)
            
        Returns:
            logits: Contrastive logits, shape (B, 1 + queue_size)
            labels: Ground truth labels (all zeros), shape (B,)
            p: Cluster assignment from query encoder, shape (B, num_classes)
        """
        # Query forward
        z_q, p = self.encoder_q(optical_q, sar_q)  # (B, projection_dim), (B, num_classes)
        
        # Key forward (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_k, _ = self.encoder_k(optical_k, sar_k)
        
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [z_q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(z_k)
        
        return logits, labels, p


def build_model(config: dict) -> UMSFNet:
    """
    Build UMSF-Net model from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: UMSF-Net model
    """
    model = UMSFNet(
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True),
        feature_dim=config.get('feature_dim', 2048),
        projection_dim=config.get('projection_dim', 256),
        num_classes=config.get('num_classes', 7),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1)
    )
    
    return model


def build_moco_model(config: dict) -> UMSFNetMoCo:
    """
    Build UMSF-Net with MoCo from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: UMSFNetMoCo model
    """
    base_model = build_model(config)
    
    model = UMSFNetMoCo(
        base_model=base_model,
        momentum=config.get('momentum', 0.999),
        queue_size=config.get('queue_size', 65536),
        temperature=config.get('temperature', 0.07)
    )
    
    return model
