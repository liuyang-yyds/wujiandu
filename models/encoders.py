"""
Encoder modules for UMSF-Net
Contains specialized encoders for optical and SAR images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple


class DespecklingModule(nn.Module):
    """
    Learnable despeckling module for SAR images
    
    Uses a lightweight CNN to reduce speckle noise while preserving
    important structural information through residual connections.
    
    Args:
        in_channels (int): Input channels. Default: 1
        hidden_channels (int): Hidden layer channels. Default: 32
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32
    ):
        super(DespecklingModule, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv3 = nn.Conv2d(
            hidden_channels, in_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: SAR image tensor, shape (B, 1, H, W)
            
        Returns:
            Despeckled image, shape (B, 1, H, W)
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio. Default: 16
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OpticalEncoder(nn.Module):
    """
    Optical image encoder using ResNet backbone
    
    Designed for multi-spectral optical remote sensing images.
    Uses ImageNet pretrained weights for better initialization.
    
    Args:
        backbone (str): Backbone type ('resnet50', 'resnet101'). Default: 'resnet50'
        pretrained (bool): Use ImageNet pretrained weights. Default: True
        out_dim (int): Output feature dimension. Default: 2048
        use_se (bool): Use SE attention blocks. Default: False
        multi_scale (bool): Return multi-scale features. Default: False
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        out_dim: int = 2048,
        use_se: bool = False,
        multi_scale: bool = False
    ):
        super(OpticalEncoder, self).__init__()
        
        self.out_dim = out_dim
        self.multi_scale = multi_scale
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # C2: 256
        self.layer2 = resnet.layer2  # C3: 512
        self.layer3 = resnet.layer3  # C4: 1024
        self.layer4 = resnet.layer4  # C5: 2048
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optional SE blocks
        if use_se:
            self.se2 = SEBlock(256)
            self.se3 = SEBlock(512)
            self.se4 = SEBlock(1024)
            self.se5 = SEBlock(2048)
        else:
            self.se2 = self.se3 = self.se4 = self.se5 = None
        
        # Projection if output dim differs
        if self.feature_dim != out_dim:
            self.fc = nn.Linear(self.feature_dim, out_dim)
        else:
            self.fc = None
            
    def forward(
        self,
        x: torch.Tensor,
        return_multiscale: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Optical image tensor, shape (B, 3, H, W)
            return_multiscale: Return multi-scale features
            
        Returns:
            features: Feature tensor, shape (B, out_dim)
            If return_multiscale: dict of multi-scale features
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Stages
        c2 = self.layer1(x)
        if self.se2:
            c2 = self.se2(c2)
            
        c3 = self.layer2(c2)
        if self.se3:
            c3 = self.se3(c3)
            
        c4 = self.layer3(c3)
        if self.se4:
            c4 = self.se4(c4)
            
        c5 = self.layer4(c4)
        if self.se5:
            c5 = self.se5(c5)
        
        # Global average pooling
        features = self.avgpool(c5)
        features = torch.flatten(features, 1)
        
        # Projection
        if self.fc is not None:
            features = self.fc(features)
        
        if return_multiscale or self.multi_scale:
            return {
                'c2': c2,  # 256
                'c3': c3,  # 512
                'c4': c4,  # 1024
                'c5': c5,  # 2048
                'features': features
            }
        
        return features


class SAREncoder(nn.Module):
    """
    SAR image encoder with learnable despeckling
    
    Designed for single-channel SAR images with inherent speckle noise.
    Includes a learnable despeckling module as preprocessing.
    
    Args:
        backbone (str): Backbone type ('resnet50', 'resnet101'). Default: 'resnet50'
        pretrained (bool): Use pretrained weights (adapted for 1-ch). Default: True
        out_dim (int): Output feature dimension. Default: 2048
        use_despeckling (bool): Use learnable despeckling module. Default: True
        use_se (bool): Use SE attention blocks. Default: False
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        out_dim: int = 2048,
        use_despeckling: bool = True,
        use_se: bool = False
    ):
        super(SAREncoder, self).__init__()
        
        self.out_dim = out_dim
        self.use_despeckling = use_despeckling
        
        # Learnable despeckling module
        if use_despeckling:
            self.despeckle = DespecklingModule(in_channels=1)
        else:
            self.despeckle = None
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv for single channel input
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize from pretrained weights (average RGB channels)
        if pretrained:
            self.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optional SE blocks
        if use_se:
            self.se2 = SEBlock(256)
            self.se3 = SEBlock(512)
            self.se4 = SEBlock(1024)
            self.se5 = SEBlock(2048)
        else:
            self.se2 = self.se3 = self.se4 = self.se5 = None
        
        # Projection if output dim differs
        if self.feature_dim != out_dim:
            self.fc = nn.Linear(self.feature_dim, out_dim)
        else:
            self.fc = None
            
    def forward(
        self,
        x: torch.Tensor,
        return_multiscale: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: SAR image tensor, shape (B, 1, H, W)
            return_multiscale: Return multi-scale features
            
        Returns:
            features: Feature tensor, shape (B, out_dim)
        """
        # Despeckling
        if self.despeckle is not None:
            x = self.despeckle(x)
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Stages
        c2 = self.layer1(x)
        if self.se2:
            c2 = self.se2(c2)
            
        c3 = self.layer2(c2)
        if self.se3:
            c3 = self.se3(c3)
            
        c4 = self.layer3(c3)
        if self.se4:
            c4 = self.se4(c4)
            
        c5 = self.layer4(c4)
        if self.se5:
            c5 = self.se5(c5)
        
        # Global average pooling
        features = self.avgpool(c5)
        features = torch.flatten(features, 1)
        
        # Projection
        if self.fc is not None:
            features = self.fc(features)
        
        if return_multiscale:
            return {
                'c2': c2,
                'c3': c3,
                'c4': c4,
                'c5': c5,
                'features': features
            }
        
        return features
