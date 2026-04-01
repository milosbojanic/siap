import torch
import torch.nn as nn

from src.config import Config


class MelanomaCNN(nn.Module):
    """
    Custom 3-layer CNN with late fusion of tabular features.
    Architecture:
      - CNN branch: 3 Conv2d layers (32->64->128) + BatchNorm + AdaptiveAvgPool
      - Fusion: concatenate CNN output (128-dim) with feature vector
      - Classifier: FC(128+feat_dim -> 64) -> FC(64 -> 1)

    Output: raw logits (no sigmoid) - use BCEWithLogitsLoss.
    """

    def __init__(self, feature_dim: int = 14):
        super().__init__()

        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Output: (batch, 128, 1, 1)
            nn.Flatten(),  # Output: (batch, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn_branch(image)
        combined = torch.cat((cnn_out, features), dim=1)
        return self.classifier(combined).squeeze(1)


class EfficientNetFusion(nn.Module):
    """
    EfficientNet B0 (pretrained on ImageNet) with late fusion of tabular features.
    Architecture:
      - Backbone: EfficientNet B0 -> 1280-dim output
      - Fusion: concatenate backbone output with feature vector
      - Classifier: FC(1280+feat_dim -> 256) -> FC(256 -> 1)

    Output: raw logits (no sigmoid) - use BCEWithLogitsLoss.
    """

    def __init__(self, feature_dim: int = 14, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained)
        backbone_out = self.backbone.classifier.in_features  # 1280
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(backbone_out + feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        backbone_out = self.backbone(image)
        combined = torch.cat((backbone_out, features), dim=1)
        return self.classifier(combined).squeeze(1)


def create_model(config: Config) -> nn.Module:
    """Factory function to create model based on config."""
    if config.model_type == "cnn":
        return MelanomaCNN(feature_dim=config.feature_dim)
    elif config.model_type == "efficientnet":
        return EfficientNetFusion(feature_dim=config.feature_dim)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
