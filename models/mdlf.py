"""
MDLF: Multimodal Deep Learning Framework for Early Alzheimer's Detection.

Full model combining:
  - MRI Branch       (EfficientNet-B4)  → F_mri  ∈ R^512
  - Cognitive Branch (3-layer MLP)      → F_cog  ∈ R^128
  - Lifestyle Branch (BiLSTM+Transformer)→ F_life ∈ R^256
  - Cross-Modal Attention Fusion         → F_fused ∈ R^512
  - Classification Head                  → P(CN, MCI, AD)

Total trainable parameters: ~24.3 million (paper §Methodology)
  - EfficientNet-B4 backbone: 17.6M (14.1M fine-tuned)
  - MLP cognitive branch:      0.8M
  - BiLSTM-Transformer:        3.2M
  - Cross-modal attention:     0.4M
  - Classification head:       2.3M
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.mri_branch import MRIBranch
from models.cognitive_branch import CognitiveBranch
from models.lifestyle_branch import LifestyleBranch
from models.fusion import CrossModalAttentionFusion


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Two-layer classification head (paper §Classification Head):
      Fully connected 512→256 (ReLU, Dropout 0.4) → 256→3 (Softmax)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, n_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Full MDLF
# ---------------------------------------------------------------------------

class MDLF(nn.Module):
    """
    Multimodal Deep Learning Framework for Alzheimer's Disease Detection.

    Parameters
    ----------
    mri_feature_dim : int           output dim of MRI branch (512)
    cog_input_dim : int             input features for cognitive branch (8)
    cog_feature_dim : int           output dim of cognitive branch (128)
    life_input_dim : int            lifestyle features per time step (8)
    life_seq_len : int              sequence length in months (12)
    life_feature_dim : int          output dim of lifestyle branch (256)
    fusion_dim : int                shared fusion / projection dim (512)
    n_classes : int                 number of diagnostic classes (3)
    pretrained_mri : bool           load ImageNet pretrained weights
    n_fine_tune_blocks : int        number of EfficientNet blocks to fine-tune (3)
    """

    def __init__(
        self,
        mri_feature_dim: int = 512,
        cog_input_dim: int = 8,
        cog_hidden_dims: list = None,
        cog_feature_dim: int = 128,
        life_input_dim: int = 8,
        life_seq_len: int = 12,
        life_feature_dim: int = 256,
        fusion_dim: int = 512,
        n_classes: int = 3,
        pretrained_mri: bool = True,
        n_fine_tune_blocks: int = 3,
        mri_dropout: float = 0.2,
        cog_dropout: float = 0.3,
        life_dropout: float = 0.1,
        cls_hidden_dim: int = 256,
        cls_dropout: float = 0.4,
    ):
        super().__init__()
        if cog_hidden_dims is None:
            cog_hidden_dims = [256, 128, 64]

        # Modality-specific encoders
        self.mri_branch = MRIBranch(
            feature_dim=mri_feature_dim,
            pretrained=pretrained_mri,
            n_fine_tune_blocks=n_fine_tune_blocks,
            dropout=mri_dropout,
        )
        self.cog_branch = CognitiveBranch(
            input_dim=cog_input_dim,
            hidden_dims=cog_hidden_dims,
            feature_dim=cog_feature_dim,
            dropout=cog_dropout,
        )
        self.life_branch = LifestyleBranch(
            input_dim=life_input_dim,
            seq_len=life_seq_len,
            bilstm_hidden=128,
            n_transformer_layers=2,
            n_heads=4,
            ff_dim=512,
            feature_dim=life_feature_dim,
            dropout=life_dropout,
        )

        # Fusion
        self.fusion = CrossModalAttentionFusion(
            mri_dim=mri_feature_dim,
            cog_dim=cog_feature_dim,
            life_dim=life_feature_dim,
            fusion_dim=fusion_dim,
            attn_dim=64,
        )

        # Classification head
        self.cls_head = ClassificationHead(
            in_dim=fusion_dim,
            hidden_dim=cls_hidden_dim,
            n_classes=n_classes,
            dropout=cls_dropout,
        )

    def forward(
        self,
        mri: torch.Tensor,
        cognitive: torch.Tensor,
        lifestyle: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mri       : (B, 3, 224, 224)
        cognitive : (B, 8)
        lifestyle : (B, 12, 8)

        Returns
        -------
        dict with keys:
          'logits'   : (B, 3)  raw class scores
          'probs'    : (B, 3)  softmax probabilities
          'alpha'    : (B, 3)  attention weights [mri, cog, life]
          'F_mri'    : (B, 512)
          'F_cog'    : (B, 128)
          'F_life'   : (B, 256)
          'F_fused'  : (B, 512)
        """
        F_mri  = self.mri_branch(mri)
        F_cog  = self.cog_branch(cognitive)
        F_life = self.life_branch(lifestyle)

        F_fused, alpha = self.fusion(F_mri, F_cog, F_life)

        logits = self.cls_head(F_fused)
        probs  = torch.softmax(logits, dim=-1)

        return {
            "logits":  logits,
            "probs":   probs,
            "alpha":   alpha,
            "F_mri":   F_mri,
            "F_cog":   F_cog,
            "F_life":  F_life,
            "F_fused": F_fused,
        }

    def predict(
        self,
        mri: torch.Tensor,
        cognitive: torch.Tensor,
        lifestyle: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted class indices."""
        out = self.forward(mri, cognitive, lifestyle)
        return out["probs"].argmax(dim=-1)

    def get_optimizer_param_groups(
        self,
        lr_main: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-5,
    ) -> list:
        """
        Return differential learning rate parameter groups (paper §Optimization):
          - EfficientNet fine-tunable layers: lr_backbone
          - All other components:             lr_main
        """
        backbone_params, _ = self.mri_branch.get_fine_tunable_params()
        backbone_ids = {id(p) for p in backbone_params}

        main_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]

        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": main_params,     "lr": lr_main,     "weight_decay": weight_decay},
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts per module."""
        def count(mod):
            return sum(p.numel() for p in mod.parameters())

        return {
            "mri_branch":  count(self.mri_branch),
            "cog_branch":  count(self.cog_branch),
            "life_branch": count(self.life_branch),
            "fusion":      count(self.fusion),
            "cls_head":    count(self.cls_head),
            "total":       count(self),
            "trainable":   sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ---------------------------------------------------------------------------
# Factory from YAML config
# ---------------------------------------------------------------------------

def build_mdlf_from_config(cfg: dict) -> MDLF:
    """Construct MDLF from a config dict (loaded from YAML)."""
    m = cfg.get("model", cfg)
    return MDLF(
        mri_feature_dim=m.get("mri_feature_dim", 512),
        cog_input_dim=m.get("cog_input_dim", 8),
        cog_hidden_dims=m.get("cog_hidden_dims", [256, 128, 64]),
        cog_feature_dim=m.get("cog_feature_dim", 128),
        life_input_dim=m.get("life_input_dim", 8),
        life_seq_len=m.get("life_seq_len", 12),
        life_feature_dim=m.get("life_feature_dim", 256),
        fusion_dim=m.get("fusion_dim", 512),
        n_classes=m.get("n_classes", 3) if "n_classes" in m else cfg.get("data", {}).get("n_classes", 3),
        pretrained_mri=m.get("mri_pretrained", True),
        n_fine_tune_blocks=3,
        cog_dropout=m.get("cog_dropout", 0.3),
        life_dropout=m.get("transformer_dropout", 0.1),
        cls_hidden_dim=m.get("cls_hidden_dim", 256),
        cls_dropout=m.get("cls_dropout", 0.4),
    )
