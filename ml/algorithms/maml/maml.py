"""
Model-Agnostic Meta-Learning (MAML / FOMAML) Implementation for MetaLearnHub.

Reference:
    Finn, Abbeel, & Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
    ICML 2017.

Key Concept:
    Unlike metric-based methods (ProtoNet, MatchingNet, RelationNet) which compute fixed distances,
    MAML is an OPTIMIZATION-BASED meta-learning algorithm.
    During adaptation on a new class with K support images, it runs an Inner Optimization Loop
    (1 to 5 gradient descent steps) to adapt parameters theta -> theta' on the support set:
        theta' = theta - alpha * grad_theta( Loss_support )
    Query images are then classified using the adapted parameters theta'.
"""

from typing import Dict, List, Tuple, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.algorithms.base import MetaLearningAlgorithm
from ml.backbones.resnet import get_backbone


class MAMLClassifierHead(nn.Module):
    """Linear adaptation head for N-way classification."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        # Initialize with small random weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MAML(nn.Module):
    """
    MAML model for vision tasks.
    Combines a deep feature extractor backbone with an inner-loop optimizable head.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        inner_lr: float = 0.05,
        inner_steps: int = 5,
    ):
        super().__init__()
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        self.embedding_dim = self.backbone.embedding_dim
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embeddings from backbone."""
        emb = self.backbone(images)
        return F.normalize(emb, p=2, dim=1)

    def adapt_head(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> nn.Linear:
        """
        Runs the MAML Inner-Loop gradient adaptation on support embeddings.

        Args:
            support_embeddings: [K*N, D]
            support_labels: [K*N]
            num_classes: N

        Returns:
            adapted_head: A trained Linear layer adapted to the support classes.
        """
        device = support_embeddings.device
        steps = steps or self.inner_steps
        lr = lr or self.inner_lr

        # Initialize adaptation head
        head = nn.Linear(self.embedding_dim, num_classes).to(device)

        # Prototype-guided weight initialization for fast convergence:
        with torch.no_grad():
            for c in range(num_classes):
                mask = (support_labels == c)
                if mask.any():
                    head.weight.data[c] = support_embeddings[mask].mean(dim=0)
            head.bias.data.zero_()

        # Inner loop optimizer
        inner_optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)

        with torch.enable_grad():
            for _ in range(steps):
                inner_optimizer.zero_grad()
                logits = head(support_embeddings)
                loss = F.cross_entropy(logits, support_labels)
                loss.backward()
                inner_optimizer.step()

        return head

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full MAML episode forward pass with inner adaptation.
        """
        num_classes = num_classes or len(torch.unique(support_labels))

        # Extract embeddings
        s_emb = self.embed(support_images)
        q_emb = self.embed(query_images)

        # Fast inner-loop adaptation on support set
        adapted_head = self.adapt_head(s_emb, support_labels, num_classes=num_classes)

        # Evaluate on query set
        q_logits = adapted_head(q_emb)
        log_probs = F.log_softmax(q_logits, dim=1)

        return log_probs, q_logits

    def predict_with_adapted_head(
        self,
        query_embedding: torch.Tensor,
        adapted_head: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference using adapted MAML head.
        """
        with torch.no_grad():
            logits = adapted_head(query_embedding)
            probs = F.softmax(logits, dim=1)[0]
            # Convert logits to distance proxy in [0, 1]
            distances = 1.0 - probs
        return probs, distances


class MAMLAlgorithm(MetaLearningAlgorithm):
    """
    MAML algorithm interface implementation for MetaLearnHub.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        inner_lr: float = 0.05,
        inner_steps: int = 5,
        device: str = "cpu",
        freeze_backbone: bool = True,
    ):
        self.device = device
        self.model = MAML(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

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

        log_probs, _ = self.model(support_images, support_labels, query_images)
        loss = F.nll_loss(log_probs, query_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = log_probs.argmax(dim=1)
        acc = (preds == query_labels).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}

    def adapt(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> nn.Linear:
        """Adapt MAML head to support images."""
        self.model.eval()
        with torch.no_grad():
            s_emb = self.model.embed(support_images.to(self.device))
        num_classes = len(torch.unique(support_labels))
        return self.model.adapt_head(s_emb, support_labels.to(self.device), num_classes=num_classes)

    def predict(
        self,
        query_images: torch.Tensor,
        adapted_state: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            q_emb = self.model.embed(query_images.to(self.device))
            logits = adapted_state(q_emb)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            return preds, confs

    def get_model(self) -> nn.Module:
        return self.model
