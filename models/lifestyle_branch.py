"""
Lifestyle Branch: BiLSTM-Transformer Hybrid.

Architecture (from paper §Lifestyle Branch):
  - Input: X_life ∈ R^(12×8) – monthly behavioral time series
  - Sinusoidal positional encoding (dim=8) added to each time step
  - BiLSTM (128 cells/direction → 256-d output at each step)
  - [CLS] token prepended to the 12-step BiLSTM output sequence
  - 2-layer Transformer Encoder (4 heads, ff_dim=512, GEGLU, pre-norm, residual)
  - Final [CLS] representation linearly projected to 256-d output
  - Output: F_life ∈ R^256
  - ~3.2 million parameters
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Learnable sinusoidal positional encoding matching paper's 8-d per step.
    Registers a (seq_len, d_model) parameter.
    """

    def __init__(self, d_model: int = 8, max_len: int = 12):
        super().__init__()
        # Compute fixed sinusoidal encoding and make it learnable
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_parameter("pe", nn.Parameter(pe))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        Returns x + positional encoding
        """
        return x + self.pe[: x.size(1), :]


# ---------------------------------------------------------------------------
# GEGLU activation (used in transformer feed-forward)
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    """Gated GELU activation: GEGLU(x) = x[:, :d] * GELU(x[:, d:])."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


# ---------------------------------------------------------------------------
# Pre-norm Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Encoder block with GEGLU feed-forward.
    4 attention heads, ff_dim=512, residual connections.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        # GEGLU: project to 2*ff_dim then gate
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim * 2),
            GEGLU(),              # output: ff_dim
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T+1, d_model)"""
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Lifestyle Branch
# ---------------------------------------------------------------------------

class LifestyleBranch(nn.Module):
    """
    BiLSTM-Transformer hybrid for temporal lifestyle feature modeling.

    Parameters
    ----------
    input_dim : int
        Number of lifestyle features per time step (default 8).
    seq_len : int
        Sequence length in months (default 12).
    bilstm_hidden : int
        Hidden units per BiLSTM direction (default 128 → 256-d bidirectional).
    n_transformer_layers : int
        Number of Transformer encoder blocks (paper: 2).
    n_heads : int
        Multi-head attention heads (paper: 4).
    ff_dim : int
        Feed-forward dimension (paper: 512).
    feature_dim : int
        Output feature dimension (default 256).
    dropout : float
        Dropout in transformer (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 8,
        seq_len: int = 12,
        bilstm_hidden: int = 128,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 512,
        feature_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        bilstm_out_dim = bilstm_hidden * 2  # 256 for bidirectional

        # Sinusoidal positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model=input_dim, max_len=seq_len)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=bilstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, bilstm_out_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=bilstm_out_dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(n_transformer_layers)
        ])

        # Final projection: 256 → feature_dim
        self.out_proj = nn.Linear(bilstm_out_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, seq_len, input_dim)

        Returns
        -------
        F_life : torch.Tensor of shape (B, feature_dim)  i.e. (B, 256)
        """
        # Step 1: Add positional encoding
        x = self.pos_enc(x)  # (B, 12, 8)

        # Step 2: BiLSTM captures local sequential dependencies
        lstm_out, _ = self.bilstm(x)  # (B, 12, 256)

        # Step 3: Prepend [CLS] token
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, 256)
        seq = torch.cat([cls, lstm_out], dim=1)   # (B, 13, 256)

        # Step 4: Transformer encoder for global self-attention
        for block in self.transformer_blocks:
            seq = block(seq)  # (B, 13, 256)

        # Step 5: Extract [CLS] representation
        cls_repr = seq[:, 0, :]   # (B, 256)

        # Step 6: Project to feature_dim
        F_life = self.out_proj(cls_repr)  # (B, 256)
        return F_life
