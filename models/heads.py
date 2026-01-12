"""
Head modules for UMSF-Net
Contains projection head and clustering head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning
    
    Projects features to a lower-dimensional space where
    contrastive loss is applied.
    
    Args:
        in_dim (int): Input feature dimension. Default: 2048
        hidden_dim (int): Hidden layer dimension. Default: 2048
        out_dim (int): Output projection dimension. Default: 256
        num_layers (int): Number of MLP layers. Default: 3
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 2048,
        out_dim: int = 256,
        num_layers: int = 3
    ):
        super(ProjectionHead, self).__init__()
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (B, in_dim)
            
        Returns:
            Projected features, shape (B, out_dim)
        """
        return self.mlp(x)


class PredictionHead(nn.Module):
    """
    Prediction head for BYOL-style asymmetric architecture
    
    Args:
        in_dim (int): Input dimension. Default: 256
        hidden_dim (int): Hidden dimension. Default: 4096
        out_dim (int): Output dimension. Default: 256
    """
    
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 4096,
        out_dim: int = 256
    ):
        super(PredictionHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ClusteringHead(nn.Module):
    """
    Clustering head for unsupervised classification
    
    Learns cluster assignments through soft assignment with
    learnable cluster centers.
    
    Args:
        in_dim (int): Input feature dimension. Default: 2048
        hidden_dim (int): Hidden layer dimension. Default: 512
        num_classes (int): Number of clusters/classes. Default: 7
        temperature (float): Temperature for softmax. Default: 0.1
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 512,
        num_classes: int = 7,
        temperature: float = 0.1
    ):
        super(ClusteringHead, self).__init__()
        
        self.temperature = temperature
        self.num_classes = num_classes
        
        # Feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Learnable cluster centers (prototypes)
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, hidden_dim)
        )
        nn.init.xavier_uniform_(self.prototypes)
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (B, in_dim)
            return_features: Whether to return intermediate features
            
        Returns:
            p: Cluster assignment probabilities, shape (B, num_classes)
        """
        # Transform features
        h = self.mlp(x)  # (B, hidden_dim)
        h = F.normalize(h, dim=1)
        
        # Normalize prototypes
        prototypes = F.normalize(self.prototypes, dim=1)
        
        # Compute similarity to prototypes
        logits = torch.mm(h, prototypes.t()) / self.temperature  # (B, num_classes)
        
        # Soft assignment
        p = F.softmax(logits, dim=1)
        
        if return_features:
            return p, h
        
        return p
    
    def get_centers(self) -> torch.Tensor:
        """Get normalized cluster centers"""
        return F.normalize(self.prototypes, dim=1)
    
    def set_temperature(self, temperature: float):
        """Update temperature for softmax"""
        self.temperature = temperature


class PrototypeClusteringHead(nn.Module):
    """
    SwAV-style prototype clustering head
    
    Uses Sinkhorn-Knopp algorithm for balanced cluster assignments.
    
    Args:
        in_dim (int): Input feature dimension. Default: 2048
        num_prototypes (int): Number of prototypes/clusters. Default: 7
        temperature (float): Temperature for softmax. Default: 0.1
        sinkhorn_iterations (int): Number of Sinkhorn iterations. Default: 3
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        num_prototypes: int = 7,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3
    ):
        super(PrototypeClusteringHead, self).__init__()
        
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.num_prototypes = num_prototypes
        
        # Prototype layer
        self.prototypes = nn.Linear(in_dim, num_prototypes, bias=False)
        
    @torch.no_grad()
    def sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply Sinkhorn-Knopp algorithm for balanced assignments
        
        Args:
            scores: Similarity scores, shape (B, num_prototypes)
            
        Returns:
            Balanced assignments, shape (B, num_prototypes)
        """
        Q = torch.exp(scores / self.temperature)
        Q = Q.t()  # (num_prototypes, B)
        
        K, B = Q.shape
        
        # Make assignments sum to 1 over samples
        sum_Q = Q.sum()
        Q /= sum_Q
        
        for _ in range(self.sinkhorn_iterations):
            # Normalize each prototype (row)
            sum_of_rows = Q.sum(dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            
            # Normalize each sample (column)
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B
        
        Q *= B  # Scale up
        return Q.t()  # (B, num_prototypes)
        
    def forward(
        self,
        x: torch.Tensor,
        use_sinkhorn: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (B, in_dim)
            use_sinkhorn: Whether to use Sinkhorn normalization
            
        Returns:
            p: Cluster assignment probabilities, shape (B, num_prototypes)
        """
        # Normalize features
        x = F.normalize(x, dim=1)
        
        # Compute scores
        scores = self.prototypes(x)  # (B, num_prototypes)
        
        if use_sinkhorn and self.training:
            return self.sinkhorn(scores)
        else:
            return F.softmax(scores / self.temperature, dim=1)
    
    def get_prototypes(self) -> torch.Tensor:
        """Get normalized prototype vectors"""
        return F.normalize(self.prototypes.weight, dim=1)
