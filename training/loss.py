"""
Loss function: Class-weighted cross-entropy for imbalanced data.

Paper §Classification Head and Loss Function (Eq. 9-10):
  L = -Σ_i w_i · y_i · log(p_i)
  w_i = N_total / (K · N_i)

For the combined dataset:
  CN (44.0%) → w_CN ≈ 0.76
  MCI (32.2%) → w_MCI ≈ 1.04
  AD (23.8%) → w_AD ≈ 1.40
"""

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(
    labels: np.ndarray,
    n_classes: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights (Eq. 10 from paper).

    Parameters
    ----------
    labels : np.ndarray of integer class indices
    n_classes : K
    device : target device for the weight tensor

    Returns
    -------
    weights : torch.Tensor of shape (n_classes,)
    """
    n_total = len(labels)
    weights = np.zeros(n_classes, dtype=np.float32)
    for k in range(n_classes):
        n_k = (labels == k).sum()
        if n_k > 0:
            weights[k] = n_total / (n_classes * n_k)
        else:
            weights[k] = 1.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Class-weighted cross-entropy loss.

    Parameters
    ----------
    class_weights : torch.Tensor (K,) or None
        If None, standard unweighted CE is used.
    label_smoothing : float
        Optional label smoothing for regularisation.
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, K)  raw class scores
        targets : (B,)    integer class indices

        Returns
        -------
        loss : scalar tensor
        """
        return self.criterion(logits, targets)
