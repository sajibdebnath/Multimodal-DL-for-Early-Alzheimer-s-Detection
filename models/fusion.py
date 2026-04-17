"""
Attention-Based Cross-Modal Fusion Module.

Architecture (from paper §Attention-Based Multimodal Fusion):
  - Projects F_mri (512), F_cog (128), F_life (256) → shared dim d_f = 512
  - Shared 2-layer attention network computes scalar energy e_i per modality
  - Softmax over {mri, cog, life} → attention weights α_i (sum to 1)
  - Fused = LayerNorm(W_proj · Σ αi·F̃i + b_proj)  ∈ R^512
  - ~0.4 million parameters (projection matrices included)

Learned per-class attention (Table in paper):
  CN:  α_mri≈0.52, α_cog≈0.33, α_life≈0.15
  MCI: α_mri≈0.38, α_cog≈0.39, α_life≈0.23
  AD:  α_mri≈0.29, α_cog≈0.44, α_life≈0.27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionFusion(nn.Module):
    """
    Attention-based cross-modal fusion for heterogeneous feature vectors.

    Parameters
    ----------
    mri_dim : int    – MRI feature dimension (512)
    cog_dim : int    – Cognitive feature dimension (128)
    life_dim : int   – Lifestyle feature dimension (256)
    fusion_dim : int – Shared projection & fusion dimension (512)
    attn_dim : int   – Attention MLP intermediate dimension (64, i.e. d_a)
    """

    def __init__(
        self,
        mri_dim: int = 512,
        cog_dim: int = 128,
        life_dim: int = 256,
        fusion_dim: int = 512,
        attn_dim: int = 64,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Modality-specific projection matrices (Eq. 4)
        self.proj_mri = nn.Linear(mri_dim, fusion_dim)
        self.proj_cog = nn.Linear(cog_dim, fusion_dim)
        self.proj_life = nn.Linear(life_dim, fusion_dim)

        # Shared 2-layer attention network (Eq. 5): W_a, b_a, w_a
        # W_a ∈ R^{64×512}, b_a ∈ R^{64}, w_a ∈ R^{64}
        self.attn_W = nn.Linear(fusion_dim, attn_dim)   # W_a · F̃_i + b_a
        self.attn_v = nn.Linear(attn_dim, 1, bias=False) # w_a^⊤ · tanh(…)

        # Final fusion projection + LayerNorm (Eq. 7)
        self.out_proj = nn.Linear(fusion_dim, fusion_dim)
        self.layer_norm = nn.LayerNorm(fusion_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        F_mri: torch.Tensor,
        F_cog: torch.Tensor,
        F_life: torch.Tensor,
    ) -> tuple:
        """
        Parameters
        ----------
        F_mri  : (B, 512)
        F_cog  : (B, 128)
        F_life : (B, 256)

        Returns
        -------
        F_fused : torch.Tensor  (B, 512)
        alpha   : torch.Tensor  (B, 3) — per-sample attention weights [mri, cog, life]
        """
        # Step 1: Project each modality to shared space (Eq. 4)
        F_mri_p  = self.proj_mri(F_mri)    # (B, 512)
        F_cog_p  = self.proj_cog(F_cog)    # (B, 512)
        F_life_p = self.proj_life(F_life)   # (B, 512)

        # Stack: (B, 3, 512)
        stacked = torch.stack([F_mri_p, F_cog_p, F_life_p], dim=1)

        # Step 2: Compute attention energies (Eq. 5)
        # e_i = w_a^⊤ · tanh(W_a · F̃_i + b_a)
        e = self.attn_v(torch.tanh(self.attn_W(stacked)))  # (B, 3, 1)
        e = e.squeeze(-1)                                    # (B, 3)

        # Step 3: Softmax → attention weights (Eq. 6)
        alpha = F.softmax(e, dim=-1)  # (B, 3)  — sums to 1

        # Step 4: Weighted sum of projected features (Eq. 7)
        alpha_expanded = alpha.unsqueeze(-1)         # (B, 3, 1)
        weighted = (stacked * alpha_expanded).sum(dim=1)  # (B, 512)

        # Step 5: Final projection + LayerNorm
        F_fused = self.layer_norm(self.out_proj(weighted))  # (B, 512)

        return F_fused, alpha

    def get_attention_weights(
        self,
        F_mri: torch.Tensor,
        F_cog: torch.Tensor,
        F_life: torch.Tensor,
    ) -> torch.Tensor:
        """Return only attention weights (no gradient tracking) for analysis."""
        with torch.no_grad():
            _, alpha = self.forward(F_mri, F_cog, F_life)
        return alpha
