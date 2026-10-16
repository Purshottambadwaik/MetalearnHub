"""
Relation Networks Implementation for MetaLearnHub.

Reference:
    Sung et al., "Learning to Compare: Relation Network for Few-Shot Learning", CVPR 2018.

Key Idea:
    Instead of handcrafting a distance metric (such as Euclidean or Cosine distance),
    Relation Networks passes the concatenation of query features and support representations
    through a learnable non-linear Relation Module (comparator):
        Relation Score = R( [f(x_query), f(x_support)] ) in [0, 1]
    where 1 indicates exact match and 0 indicates unrelated.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.algorithms.base import MetaLearningAlgorithm
from ml.backbones.resnet import get_backbone


class RelationModule(nn.Module):
    """
    Deep non-linear comparator module.
    Takes query and prototype embeddings and outputs a relation score in [0, 1].
    Uses relational representations: [q * p, |q - p|] (element-wise correlation & difference).
    """

    def __init__(self, embedding_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, q: torch.Tensor, p: Optional[torch.Tensor] = None) -> torch.Tensor:
        if p is not None:
            diff = torch.abs(q - p)
            prod = q * p
            x = torch.cat([prod, diff], dim=-1)
        else:
            d = q.size(-1) // 2
            q_part = q[..., :d]
            p_part = q[..., d:]
            diff = torch.abs(q_part - p_part)
            prod = q_part * p_part
            x = torch.cat([prod, diff], dim=-1)
        return self.net(x).squeeze(-1)


class RelationNetwork(nn.Module):
    """
    Relation Network model.
    Feature Extractor Backbone + Learnable Non-linear Relation Module.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        self.embedding_dim = self.backbone.embedding_dim
        self.relation_module = RelationModule(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
        )

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Embed images into feature space."""
        return self.backbone(images)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single episode.
        Computes prototypes for classes, concatenates with queries, and scores relations.
        """
        n_support = support_images.size(0)
        n_query = query_images.size(0)

        all_images = torch.cat([support_images, query_images], dim=0)
        all_embeddings = self.backbone(all_images)
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

        support_emb = all_embeddings[:n_support]
        query_emb = all_embeddings[n_support:]

        if num_classes is None:
            num_classes = len(torch.unique(support_labels))

        # Compute prototypes
        prototypes = []
        for c in range(num_classes):
            mask = (support_labels == c)
            class_proto = support_emb[mask].mean(dim=0, keepdim=True)
            class_proto = F.normalize(class_proto, p=2, dim=1)
            prototypes.append(class_proto)
        prototypes = torch.cat(prototypes, dim=0)  # [N, D]

        # Combine each query with each class prototype: [Q, N, 2*D]
        q_rep = query_emb.unsqueeze(1).repeat(1, num_classes, 1)  # [Q, N, D]
        p_rep = prototypes.unsqueeze(0).repeat(n_query, 1, 1)    # [Q, N, D]
        relation_pairs = torch.cat([q_rep, p_rep], dim=2)        # [Q, N, 2*D]

        # Pass through relation module: [Q, N]
        flat_pairs = relation_pairs.view(-1, self.embedding_dim * 2)
        relation_scores = self.relation_module(flat_pairs).view(n_query, num_classes)

        return relation_scores, prototypes

    def predict_with_prototypes(
        self,
        query_embedding: torch.Tensor,
        prototypes: torch.Tensor,
        temperature: float = 16.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict relation scores against registered prototypes.

        Args:
            query_embedding: [1, D]
            prototypes: [N, D]

        Returns:
            class_probs: [N]
            distances: [N] (1.0 - relation_score)
        """
        device = query_embedding.device
        num_classes = prototypes.size(0)

        q_rep = query_embedding.repeat(num_classes, 1)  # [N, D]
        pairs = torch.cat([q_rep, prototypes.to(device)], dim=1)  # [N, 2*D]

        relation_scores = self.relation_module(pairs)  # [N] in [0, 1]
        distances = 1.0 - relation_scores

        # Temperature-scaled softmax for discriminative class probabilities
        probs = F.softmax(relation_scores * temperature, dim=0)
        return probs, distances


class RelationNetworkAlgorithm(MetaLearningAlgorithm):
    """
    Relation Network algorithm wrapper adhering to MetaLearningAlgorithm.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        freeze_backbone: bool = False,
    ):
        self.device = device
        self.model = RelationNetwork(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = None

    def meta_train_step(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        num_classes = len(torch.unique(support_labels))
        scores, _ = self.model(support_images, support_labels, query_images, num_classes=num_classes)

        # MSE loss on relation scores (1 for ground truth class, 0 for others)
        one_hot_targets = F.one_hot(query_labels, num_classes=num_classes).float()
        loss = F.mse_loss(scores, one_hot_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = scores.argmax(dim=1)
        acc = (preds == query_labels).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}

    def adapt(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            emb = self.model.embed(support_images.to(self.device))
            emb = F.normalize(emb, p=2, dim=1)
            num_classes = len(torch.unique(support_labels))
            prototypes = []
            for c in range(num_classes):
                mask = (support_labels == c)
                prototypes.append(emb[mask].mean(dim=0, keepdim=True))
            prototypes = torch.cat(prototypes, dim=0)
            prototypes = F.normalize(prototypes, p=2, dim=1)
        return {"prototypes": prototypes}

    def predict(
        self,
        query_images: torch.Tensor,
        adapted_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            q_emb = self.model.embed(query_images.to(self.device))
            q_emb = F.normalize(q_emb, p=2, dim=1)
            protos = adapted_state["prototypes"].to(self.device)

            probs, _ = self.model.predict_with_prototypes(q_emb[0:1], protos)
            conf, pred = probs.max(dim=0)
            return pred.unsqueeze(0), conf.unsqueeze(0)

    def get_model(self) -> nn.Module:
        return self.model
