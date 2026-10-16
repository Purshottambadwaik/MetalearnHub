"""
Matching Networks Implementation for MetaLearnHub.

Reference:
    Vinyals et al., "Matching Networks for One Shot Learning", NeurIPS 2016.

Key Idea:
    Instead of collapsing all support examples of a class into a single average centroid
    (like Prototypical Networks), Matching Networks compares the query against EVERY individual
    support exemplar using an attention mechanism:
        P(y = c | x_hat) = sum_{i in S_c} a(x_hat, x_i)
    where a(x_hat, x_i) = softmax(cosine_similarity(f(x_hat), g(x_i)) * tau).
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.algorithms.base import MetaLearningAlgorithm
from ml.backbones.resnet import get_backbone


class MatchingNetwork(nn.Module):
    """
    Matching Network model.
    Maps images to an embedding space and classifies queries using attention over support exemplars.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        temperature: float = 16.0,
    ):
        super().__init__()
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        self.embedding_dim = self.backbone.embedding_dim
        self.temperature = temperature

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

        Args:
            support_images: [N*K, 3, H, W]
            support_labels: [N*K] - integer class labels (0 to N-1)
            query_images: [N*Q, 3, H, W]
            num_classes: Number of unique classes N

        Returns:
            log_probabilities: [N*Q, N]
            attention_weights: [N*Q, N*K]
        """
        all_images = torch.cat([support_images, query_images], dim=0)
        all_embeddings = self.backbone(all_images)
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

        n_support = support_images.size(0)
        support_emb = all_embeddings[:n_support]
        query_emb = all_embeddings[n_support:]

        if num_classes is None:
            num_classes = len(torch.unique(support_labels))

        # Cosine similarity matrix: [N*Q, N*K]
        sims = torch.mm(query_emb, support_emb.t()) * self.temperature
        attn_weights = F.softmax(sims, dim=1)  # [N*Q, N*K]

        # One-hot encode support labels: [N*K, N]
        support_one_hot = F.one_hot(support_labels, num_classes=num_classes).float()

        # Class probabilities: [N*Q, N] = [N*Q, N*K] x [N*K, N]
        probs = torch.mm(attn_weights, support_one_hot)
        probs = torch.clamp(probs, min=1e-8)
        log_probs = torch.log(probs)

        return log_probs, attn_weights

    def predict_with_support_embeddings(
        self,
        query_embedding: torch.Tensor,
        support_embeddings_list: List[torch.Tensor],
        temperature: float = 16.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference function using precomputed normalized support embeddings.

        Args:
            query_embedding: [1, D] (L2 normalized)
            support_embeddings_list: list of [K_c, D] tensors, one per class

        Returns:
            class_probs: [N] class probabilities
            effective_distances: [N] cosine distance to nearest exemplar or weighted distance
        """
        device = query_embedding.device
        all_supports = []
        labels = []

        for class_idx, class_supp in enumerate(support_embeddings_list):
            class_supp = class_supp.to(device)
            class_supp = F.normalize(class_supp, p=2, dim=1)
            all_supports.append(class_supp)
            labels.extend([class_idx] * class_supp.size(0))

        stacked_supports = torch.cat(all_supports, dim=0)  # [Total_K, D]
        support_labels = torch.tensor(labels, dtype=torch.long, device=device)
        num_classes = len(support_embeddings_list)

        # Similarities across all exemplars: [1, Total_K]
        sims = torch.mm(query_embedding, stacked_supports.t())
        dists = 1.0 - sims  # Cosine distances to each exemplar

        # Softmax attention
        attn = F.softmax(sims * temperature, dim=1)

        # One-hot aggregation per class: [Total_K, N]
        one_hot = F.one_hot(support_labels, num_classes=num_classes).float()
        class_probs = torch.mm(attn, one_hot)[0]  # [N]

        # Calculate minimum distance per class (nearest exemplar distance)
        class_min_dists = torch.zeros(num_classes, device=device)
        for class_idx in range(num_classes):
            mask = (support_labels == class_idx)
            class_min_dists[class_idx] = dists[0, mask].min()

        return class_probs, class_min_dists


class MatchingNetworkAlgorithm(MetaLearningAlgorithm):
    """
    Matching Network algorithm implementation adhering to MetaLearningAlgorithm.
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
        self.model = MatchingNetwork(
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
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            emb = self.model.embed(support_images.to(self.device))
            emb = F.normalize(emb, p=2, dim=1)
        return {"support_embeddings": emb, "support_labels": support_labels}

    def predict(
        self,
        query_images: torch.Tensor,
        adapted_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            q_emb = self.model.embed(query_images.to(self.device))
            q_emb = F.normalize(q_emb, p=2, dim=1)
            s_emb = adapted_state["support_embeddings"].to(self.device)
            s_labels = adapted_state["support_labels"].to(self.device)

            sims = torch.mm(q_emb, s_emb.t()) * self.model.temperature
            attn = F.softmax(sims, dim=1)
            num_classes = len(torch.unique(s_labels))
            one_hot = F.one_hot(s_labels, num_classes=num_classes).float()
            probs = torch.mm(attn, one_hot)
            confs, preds = probs.max(dim=1)
            return preds, confs

    def get_model(self) -> nn.Module:
        return self.model
