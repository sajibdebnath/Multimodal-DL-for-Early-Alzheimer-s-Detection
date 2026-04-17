"""
MRI Branch: EfficientNet-B4 with Transfer Learning.

Architecture (from paper §Modality-Specific Feature Extraction):
  - EfficientNet-B4 pretrained on ImageNet
  - Blocks 0-4 frozen; blocks 5-7 fine-tuned at lr=1e-5
  - Final FC replaced by GAP → 512-d linear projection with ReLU + BN
  - Output: F_mri ∈ R^512
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class MRIBranch(nn.Module):
    """
    EfficientNet-B4 encoder for structural MRI slices.

    Parameters
    ----------
    feature_dim : int
        Output feature dimension (default: 512, per paper).
    pretrained : bool
        Load ImageNet pretrained weights.
    n_fine_tune_blocks : int
        Number of MBConv blocks to unfreeze from the end (paper: 3).
    dropout : float
        Dropout rate before projection.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = True,
        n_fine_tune_blocks: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                "efficientnet_b4",
                pretrained=pretrained,
                num_classes=0,       # remove classification head
                global_pool="avg",   # GAP built in
            )
            penultimate_dim = self.backbone.num_features  # 1792 for B4
        else:
            # Fallback: use torchvision EfficientNet_B4
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b4(weights=weights)
            # Remove the classifier
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            penultimate_dim = 1792
            self._use_torchvision = True

        self._use_torchvision = not TIMM_AVAILABLE

        # Projection head: 1792 → feature_dim
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Sequential(
            nn.Linear(penultimate_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Freeze / unfreeze according to paper
        self._set_frozen_layers(n_fine_tune_blocks)

    def _set_frozen_layers(self, n_fine_tune_blocks: int) -> None:
        """Freeze all backbone parameters except the last n blocks."""
        # First, freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        if TIMM_AVAILABLE:
            # timm EfficientNet: blocks are in backbone.blocks (list of stages)
            blocks = list(self.backbone.blocks)
            total = len(blocks)
            # Unfreeze last n_fine_tune_blocks stages + conv_head
            for block in blocks[max(0, total - n_fine_tune_blocks):]:
                for param in block.parameters():
                    param.requires_grad = True
            # Always unfreeze the final conv
            if hasattr(self.backbone, "conv_head"):
                for param in self.backbone.conv_head.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "bn2"):
                for param in self.backbone.bn2.parameters():
                    param.requires_grad = True
        else:
            # torchvision: children are features, avgpool, classifier
            features = list(self.backbone.children())
            for layer in features[max(0, len(features) - n_fine_tune_blocks):]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, 3, 224, 224)

        Returns
        -------
        F_mri : torch.Tensor of shape (B, feature_dim)
        """
        if self._use_torchvision:
            # Manual GAP after feature extraction
            feat = self.backbone(x)             # (B, 1792, h, w)
            feat = feat.mean(dim=[-2, -1])       # GAP → (B, 1792)
        else:
            feat = self.backbone(x)              # (B, 1792) timm does GAP internally

        feat = self.dropout(feat)
        F_mri = self.projection(feat)            # (B, 512)
        return F_mri

    def get_fine_tunable_params(self):
        """Return parameter groups for differential learning rates."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.projection.parameters()) + list(self.dropout.parameters())
        return backbone_params, head_params
