"""
Cognitive Branch: Multi-Layer Perceptron.

Architecture (from paper §Cognitive Branch):
  - Input: 8-dim feature vector (MMSE, CDR, CDR-SB, CDR-M, CDR-O, age, sex, edu)
  - 3-layer MLP with hidden dims [256, 128, 64]
  - ReLU + BatchNorm after each linear; Dropout(0.3) after each non-linearity
  - Final linear projects 64 → 128
  - Output: F_cog ∈ R^128
  - ~0.8 million parameters
"""

import torch
import torch.nn as nn


class CognitiveBranch(nn.Module):
    """
    Three-layer MLP encoder for cognitive / demographic features.

    Parameters
    ----------
    input_dim : int
        Number of cognitive + demographic features (default 8).
    hidden_dims : list[int]
        Hidden layer sizes (paper: [256, 128, 64]).
    feature_dim : int
        Output feature dimension (default 128).
    dropout : float
        Dropout probability (paper: 0.3).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list = None,
        feature_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.feature_dim = feature_dim
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        # Final projection: 64 → 128
        self.out_proj = nn.Linear(in_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_dim)

        Returns
        -------
        F_cog : torch.Tensor of shape (B, feature_dim)  i.e. (B, 128)
        """
        h = self.mlp(x)           # (B, 64)
        F_cog = self.out_proj(h)  # (B, 128)
        return F_cog
