"""
Evaluation metrics for UMSF-Net
Contains clustering and classification metrics
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score as sklearn_f1_score,
    accuracy_score
)
from typing import Optional


class ClusteringMetrics:
    """
    Metrics for evaluating clustering quality
    
    Includes:
    - Clustering Accuracy (ACC)
    - Normalized Mutual Information (NMI)
    - Adjusted Rand Index (ARI)
    - F1 Score
    """
    
    @staticmethod
    def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute clustering accuracy using Hungarian algorithm
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            Clustering accuracy (ACC)
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        
        assert y_true.shape == y_pred.shape, "Shape mismatch"
        
        # Build confusion matrix
        n_clusters = max(y_pred.max(), y_true.max()) + 1
        contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        
        for i in range(len(y_true)):
            contingency[y_pred[i], y_true[i]] += 1
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-contingency)
        
        # Compute accuracy
        accuracy = contingency[row_ind, col_ind].sum() / len(y_true)
        
        return accuracy
    
    @staticmethod
    def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Normalized Mutual Information
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            NMI score
        """
        return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    
    @staticmethod
    def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Adjusted Rand Index
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            ARI score
        """
        return adjusted_rand_score(y_true, y_pred)
    
    @staticmethod
    def f1_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> np.ndarray:
        """
        Compute F1 score with Hungarian matching
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            average: Averaging method ('macro', 'weighted', None)
            
        Returns:
            F1 score
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        
        # Find best matching using Hungarian algorithm
        n_clusters = max(y_pred.max(), y_true.max()) + 1
        contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        
        for i in range(len(y_true)):
            contingency[y_pred[i], y_true[i]] += 1
        
        row_ind, col_ind = linear_sum_assignment(-contingency)
        
        # Remap predictions
        mapping = {r: c for r, c in zip(row_ind, col_ind)}
        y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
        
        return sklearn_f1_score(y_true, y_pred_mapped, average=average, zero_division=0)
    
    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix with Hungarian matching
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            normalize: Normalization method ('true', 'pred', 'all', None)
            
        Returns:
            Confusion matrix
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        
        # Hungarian matching
        n_clusters = max(y_pred.max(), y_true.max()) + 1
        contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        
        for i in range(len(y_true)):
            contingency[y_pred[i], y_true[i]] += 1
        
        row_ind, col_ind = linear_sum_assignment(-contingency)
        mapping = {r: c for r, c in zip(row_ind, col_ind)}
        y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
        
        return confusion_matrix(y_true, y_pred_mapped, normalize=normalize)


def entropy(p: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute entropy of probability distribution
    
    Args:
        p: Probability distribution
        eps: Small constant for numerical stability
        
    Returns:
        Entropy value
    """
    p = np.asarray(p)
    p = p / (p.sum() + eps)
    return -np.sum(p * np.log(p + eps))


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering purity
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        
    Returns:
        Purity score
    """
    contingency = confusion_matrix(y_pred, y_true)
    return np.sum(np.max(contingency, axis=1)) / len(y_true)
