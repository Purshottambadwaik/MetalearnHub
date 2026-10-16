"""
Train Relation Networks Comparator Module on Animal Dataset Embeddings.
Solves the flat-probability collapse by training the RelationModule
with relational difference features and balanced binary cross-entropy.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import CHECKPOINTS_DIR
from ml.algorithms.relation_networks.relation_net import RelationModule
from ml.inference.predictor import FewShotPredictor
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm


def train_comparator(epochs: int = 60, lr: float = 1e-3):
    print("\n" + "=" * 65)
    print(" Training Relation Networks Comparator on Animal Classes")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    algo = PrototypicalNetworkAlgorithm(backbone_name="resnet50", pretrained=True, device=device)
    pred = FewShotPredictor(algo)

    ready_classes = [c for c in pred.classes.values() if c.is_ready and c.support_embeddings is not None]
    print(f"Loaded {len(ready_classes)} classes from registry.")

    # Build balanced pairs (Positives: 1.0, Negatives: 0.0)
    X_q = []
    X_p = []
    Y = []

    for i, c1 in enumerate(ready_classes):
        embs1 = c1.support_embeddings.to(device)
        embs1 = F.normalize(embs1, p=2, dim=1)
        p1 = c1.prototype.to(device)
        p1 = F.normalize(p1, p=2, dim=1)

        # Positive pairs (query from c1 with c1 prototype)
        for e in embs1:
            X_q.append(e)
            X_p.append(p1[0])
            Y.append(1.0)

        # Negative pairs (query from c1 with other prototypes)
        for j, c2 in enumerate(ready_classes):
            if i != j:
                p2 = c2.prototype.to(device)
                p2 = F.normalize(p2, p=2, dim=1)
                for e in embs1[:2]:  # Take 2 negative exemplars per competing class
                    X_q.append(e)
                    X_p.append(p2[0])
                    Y.append(0.0)

    X_q = torch.stack(X_q)
    X_p = torch.stack(X_p)
    Y = torch.tensor(Y, dtype=torch.float, device=device)

    num_pos = int((Y == 1.0).sum().item())
    num_neg = int((Y == 0.0).sum().item())
    print(f"Dataset generated: {len(Y)} total pairs ({num_pos} positive, {num_neg} negative)")

    # Weight positive loss to balance precision
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = RelationModule(embedding_dim=2048, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Compute relational features
        diff = torch.abs(X_q - X_p)
        prod = X_q * X_p
        feat = torch.cat([prod, diff], dim=-1)

        # Get raw logits (before final sigmoid)
        logits = model.net[:-1](feat).squeeze(-1)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                scores = torch.sigmoid(logits)
                preds = (scores > 0.5).float()
                acc = (preds == Y).float().mean().item()
                mean_pos_score = scores[Y == 1.0].mean().item()
                mean_neg_score = scores[Y == 0.0].mean().item()
            print(
                f" Epoch {epoch:2d}/{epochs:2d} | Loss: {loss.item():.4f} | "
                f"Pair Acc: {acc*100:5.1f}% | "
                f"Avg Pos Score: {mean_pos_score:.3f} | Avg Neg Score: {mean_neg_score:.3f}"
            )

    elapsed = time.time() - t0
    ckpt_dir = CHECKPOINTS_DIR / "relation_networks"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_relation_model.pt"

    torch.save({
        "relation_module_state_dict": model.state_dict(),
        "algorithm": "relation_networks",
        "embedding_dim": 2048,
        "epochs": epochs,
    }, ckpt_path)

    print("\n" + "=" * 65)
    print(f" Training Complete in {elapsed:.2f}s!")
    print(f" Relation Model Checkpoint saved to: {ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    train_comparator()
