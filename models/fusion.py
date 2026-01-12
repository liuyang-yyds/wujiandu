"""
Fusion modules for UMSF-Net
Contains cross-modal attention fusion mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion Module
    
    Implements multi-head cross-attention to learn adaptive fusion weights
    between optical and SAR features.
    
    Args:
        in_dim (int): Input feature dimension. Default: 2048
        num_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.1
        use_layer_norm (bool): Use layer normalization. Default: True
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super(CrossModalAttentionFusion, self).__init__()
        
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for optical modality
        self.q_optical = nn.Linear(in_dim, in_dim)
        self.k_optical = nn.Linear(in_dim, in_dim)
        self.v_optical = nn.Linear(in_dim, in_dim)
        
        # Query, Key, Value projections for SAR modality
        self.q_sar = nn.Linear(in_dim, in_dim)
        self.k_sar = nn.Linear(in_dim, in_dim)
        self.v_sar = nn.Linear(in_dim, in_dim)
        
        # Output projections
        self.out_proj_optical = nn.Linear(in_dim, in_dim)
        self.out_proj_sar = nn.Linear(in_dim, in_dim)
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm_optical = nn.LayerNorm(in_dim)
            self.layer_norm_sar = nn.LayerNorm(in_dim)
            self.layer_norm_out = nn.LayerNorm(in_dim)
        else:
            self.layer_norm_optical = None
            self.layer_norm_sar = None
            self.layer_norm_out = None
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
        
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, num_heads, 1, head_dim)
    
    def _cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-modal attention
        
        Args:
            query: Query tensor, shape (B, num_heads, 1, head_dim)
            key: Key tensor, shape (B, num_heads, 1, head_dim)
            value: Value tensor, shape (B, num_heads, 1, head_dim)
            
        Returns:
            output: Attended features, shape (B, in_dim)
            attention_weights: Attention weights
        """
        batch_size = query.size(0)
        
        # Attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, value)
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1)
        
        return out, attn_weights
        
    def forward(
        self,
        f_optical: torch.Tensor,
        f_sar: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of cross-modal attention fusion
        
        Args:
            f_optical: Optical features, shape (B, in_dim)
            f_sar: SAR features, shape (B, in_dim)
            
        Returns:
            f_fused: Fused features, shape (B, in_dim)
            attention_info: Dictionary containing attention weights
        """
        # Layer normalization
        if self.layer_norm_optical is not None:
            f_optical_norm = self.layer_norm_optical(f_optical)
            f_sar_norm = self.layer_norm_sar(f_sar)
        else:
            f_optical_norm = f_optical
            f_sar_norm = f_sar
        
        # Cross-attention: Optical attends to SAR
        q_opt = self._reshape_for_attention(self.q_optical(f_optical_norm))
        k_sar = self._reshape_for_attention(self.k_sar(f_sar_norm))
        v_sar = self._reshape_for_attention(self.v_sar(f_sar_norm))
        
        opt_attended, attn_opt2sar = self._cross_attention(q_opt, k_sar, v_sar)
        opt_attended = self.out_proj_optical(opt_attended)
        opt_attended = f_optical + self.dropout(opt_attended)  # Residual
        
        # Cross-attention: SAR attends to Optical
        q_sar = self._reshape_for_attention(self.q_sar(f_sar_norm))
        k_opt = self._reshape_for_attention(self.k_optical(f_optical_norm))
        v_opt = self._reshape_for_attention(self.v_optical(f_optical_norm))
        
        sar_attended, attn_sar2opt = self._cross_attention(q_sar, k_opt, v_opt)
        sar_attended = self.out_proj_sar(sar_attended)
        sar_attended = f_sar + self.dropout(sar_attended)  # Residual
        
        # Gated fusion
        concat_features = torch.cat([opt_attended, sar_attended], dim=-1)
        gate_weights = self.gate(concat_features)  # (B, 2)
        
        # Weighted fusion
        f_fused = (
            gate_weights[:, 0:1] * opt_attended +
            gate_weights[:, 1:2] * sar_attended
        )
        
        # Output layer norm
        if self.layer_norm_out is not None:
            f_fused = self.layer_norm_out(f_fused)
        
        attention_info = {
            'optical_to_sar': attn_opt2sar,
            'sar_to_optical': attn_sar2opt,
            'gate_weights': gate_weights
        }
        
        return f_fused, attention_info


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion baseline
    
    Args:
        in_dim (int): Input feature dimension per modality
        out_dim (int): Output feature dimension
    """
    
    def __init__(self, in_dim: int = 2048, out_dim: int = 2048):
        super(ConcatFusion, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        f_optical: torch.Tensor,
        f_sar: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        concat = torch.cat([f_optical, f_sar], dim=-1)
        return self.fc(concat), None


class AddFusion(nn.Module):
    """
    Element-wise addition fusion baseline
    
    Args:
        in_dim (int): Feature dimension
    """
    
    def __init__(self, in_dim: int = 2048):
        super(AddFusion, self).__init__()
        
        self.layer_norm = nn.LayerNorm(in_dim)
        
    def forward(
        self,
        f_optical: torch.Tensor,
        f_sar: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        fused = f_optical + f_sar
        return self.layer_norm(fused), None


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion with FPN-style lateral connections
    
    Args:
        feature_dims (list): Feature dimensions at each scale. Default: [256, 512, 1024, 2048]
        out_dim (int): Output feature dimension. Default: 2048
    """
    
    def __init__(
        self,
        feature_dims: list = [256, 512, 1024, 2048],
        out_dim: int = 2048
    ):
        super(MultiScaleFusion, self).__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, 256, kernel_size=1)
            for dim in feature_dims
        ])
        
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
            for _ in feature_dims
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * len(feature_dims), out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(
        self,
        features: dict
    ) -> torch.Tensor:
        """
        Args:
            features: Dict with keys 'c2', 'c3', 'c4', 'c5'
            
        Returns:
            Fused features, shape (B, out_dim)
        """
        feature_list = [features['c2'], features['c3'], features['c4'], features['c5']]
        
        # Lateral connections
        laterals = [
            conv(f) for conv, f in zip(self.lateral_convs, feature_list)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )
        
        # Smooth
        outputs = [
            conv(f) for conv, f in zip(self.smooth_convs, laterals)
        ]
        
        # Upsample all to the same size and concatenate
        target_size = outputs[0].shape[2:]
        outputs = [
            F.interpolate(f, size=target_size, mode='nearest')
            for f in outputs
        ]
        
        concat = torch.cat(outputs, dim=1)
        fused = self.fusion_conv(concat)
        
        # Global average pooling
        out = self.avgpool(fused)
        out = torch.flatten(out, 1)
        
        return out
