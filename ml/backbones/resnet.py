"""
ResNet-18 Backbone for MetaLearnHub.

Provides the visual feature extractor used by meta-learning algorithms.
Outputs a fixed-dimension embedding vector for each input image.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 feature extractor for meta-learning.
    
    Removes the final classification layer and outputs a 512-dimensional
    embedding vector for each input image.
    
    Architecture:
        Image [B, 3, 224, 224]
            ↓
        ResNet-18 conv layers
            ↓
        Global Average Pooling
            ↓
        Embedding [B, 512]
    """
    
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            freeze: Whether to freeze backbone parameters
        """
        super().__init__()
        
        # Load ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)
        
        # Remove the final FC layer — keep everything up to avgpool
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = resnet.avgpool
        self.embedding_dim = 512  # ResNet-18 final layer channels
        
        # Optionally freeze
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from input images.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Embedding tensor [B, 512]
        """
        features = self.features(x)
        pooled = self.avgpool(features)
        embedding = pooled.view(pooled.size(0), -1)  # [B, 512]
        return embedding


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 feature extractor for meta-learning.
    Outputs a high-capacity 2048-dimensional embedding vector for each input image.
    """
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
        
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = resnet.avgpool
        self.embedding_dim = 2048
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.avgpool(features)
        return pooled.view(pooled.size(0), -1)


def get_backbone(name: str = "resnet50", pretrained: bool = True, freeze: bool = False) -> nn.Module:
    """
    Factory function for backbone creation.
    
    Args:
        name: Backbone name ('resnet18', 'resnet50')
        pretrained: Whether to use pretrained weights
        freeze: Whether to freeze parameters
    
    Returns:
        Backbone module
    """
    if name == "resnet18":
        return ResNet18Backbone(pretrained=pretrained, freeze=freeze)
    elif name == "resnet50":
        return ResNet50Backbone(pretrained=pretrained, freeze=freeze)
    else:
        raise ValueError(f"Unknown backbone: {name}. Available: ['resnet18', 'resnet50']")
