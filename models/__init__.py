"""
UMSF-Net Model Components
"""

from .umsf_net import UMSFNet, UMSFNetMoCo, build_model, build_moco_model
from .encoders import OpticalEncoder, SAREncoder, DespecklingModule, SEBlock
from .fusion import CrossModalAttentionFusion, ConcatFusion, AddFusion, MultiScaleFusion
from .heads import ProjectionHead, ClusteringHead, PredictionHead, PrototypeClusteringHead
from .losses import (
    InfoNCELoss,
    NTXentLoss,
    CrossModalContrastiveLoss,
    ClusteringLoss,
    ConsistencyLoss,
    UMSFLoss
)

__all__ = [
    # Main network
    'UMSFNet',
    'UMSFNetMoCo',
    'build_model',
    'build_moco_model',
    # Encoders
    'OpticalEncoder',
    'SAREncoder',
    'DespecklingModule',
    'SEBlock',
    # Fusion
    'CrossModalAttentionFusion',
    'ConcatFusion',
    'AddFusion',
    'MultiScaleFusion',
    # Heads
    'ProjectionHead',
    'ClusteringHead',
    'PredictionHead',
    'PrototypeClusteringHead',
    # Losses
    'InfoNCELoss',
    'NTXentLoss',
    'CrossModalContrastiveLoss',
    'ClusteringLoss',
    'ConsistencyLoss',
    'UMSFLoss',
]
