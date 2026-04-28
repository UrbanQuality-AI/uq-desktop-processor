"""
Fine-tuned vision model construction and forward pass utilities for image scoring.
"""

import logging
from typing import Any, cast

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

try:
    import timm
except Exception as import_error:
    raise RuntimeError("Install timm: pip install timm") from import_error


class ViTMultiHead(nn.Module):
    """
    ViT backbone with a multi-output prediction head.

    Example::
        In: ViTMultiHead(...)
        Out: initialized instance ready for use
    """

    def __init__(self, model_name: str, num_outputs: int = 6, image_size: int = 224) -> None:
        """
        Create model backbone and MLP head for fine-tuned scoring.

        :param model_name: timm model name for the backbone.
        :param num_outputs: Number of output targets.
        :param image_size: Input image size used by the backbone.

        Example::
            In: ViTMultiHead("vit_base_patch14_dinov2.lvd142m", num_outputs=6, image_size=224)
            Out: initialized module ready for forward passes
        """
        super().__init__()
        log.info("Initializing ViTMultiHead. Backbone: '%s'", model_name)

        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, dynamic_img_size=True, img_size=image_size
        )
        backbone_any = cast(Any, self.backbone)

        if hasattr(backbone_any, "num_features"):
            feat_dim = int(backbone_any.num_features)
        else:
            feat_dim = int(backbone_any.embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass.

        :param x: Input batch tensor ``(N, C, H, W)``.
        :return: Predicted outputs ``(N, num_outputs)``.

        Example::
            In: model.forward(torch.randn(2, 3, 224, 224))
            Out: tensor with shape (2, num_outputs)
        """
        # Combine global CLS token with pooled patch context before the MLP head.
        backbone_any = cast(Any, self.backbone)
        features = backbone_any.forward_features(x)
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
        pooled_patches = torch.mean(patch_tokens, dim=1)

        combined = torch.cat([cls_token, pooled_patches], dim=1)
        return self.head(combined)
