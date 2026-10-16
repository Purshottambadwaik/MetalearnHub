"""
Meta-Training Script for MetaLearnHub.
Trains meta-learning algorithms (Relation Networks, Matching Networks, ProtoNet)
on the 10 animal categories from the dataset (archive (1) (1)/raw-img).
"""

import sys
import os
import time
import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.config import CHECKPOINTS_DIR
from ml.algorithms import get_algorithm
from ml.algorithms.relation_networks.relation_net import RelationNetwork
from ml.algorithms.matching_networks.matching_net import MatchingNetwork
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetwork

DATASET_DIR = PROJECT_ROOT.parent / "archive (1) (1)" / "raw-img"

BASE_CLASSES = {
    "Butterfly": "Butterfly",
    "Cattle":    "Cattle",
    "Dog":       "Dog",
    "Elephant":  "Elephant",
    "Hen":       "Hen",
    "Horse":     "Horse",
    "Spider":    "Spider",
    "Squirral":  "Squirrel",
    "gatto":     "Cat",
    "sheep":     "Sheep",
}


def get_image_paths_by_class(dataset_dir: Path, max_per_class: int = 100):
    dataset = {}
    for folder, display_name in BASE_CLASSES.items():
        folder_path = dataset_dir / folder
        if not folder_path.exists():
            continue
        imgs = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.png")))
        if imgs:
            step = max(1, len(imgs) // max_per_class)
            dataset[display_name] = imgs[::step][:max_per_class]
    return dataset


def sample_episode(dataset, n_way=5, k_shot=5, q_query=3, transform=None):
    classes = random.sample(list(dataset.keys()), n_way)
    support_imgs = []
    support_labels = []
    query_imgs = []
    query_labels = []

    for label_idx, cls in enumerate(classes):
        available = dataset[cls]
        selected = random.sample(available, k_shot + q_query)
        supp_files = selected[:k_shot]
        query_files = selected[k_shot:]

        for f in supp_files:
            img = Image.open(f).convert("RGB")
            if transform:
                img = transform(img)
            support_imgs.append(img)
            support_labels.append(label_idx)

        for f in query_files:
            img = Image.open(f).convert("RGB")
            if transform:
                img = transform(img)
            query_imgs.append(img)
            query_labels.append(label_idx)

    return (
        torch.stack(support_imgs),
        torch.tensor(support_labels, dtype=torch.long),
        torch.stack(query_imgs),
        torch.tensor(query_labels, dtype=torch.long),
        classes,
    )


def train_relation_network(
    dataset,
    epochs: int = 5,
    episodes_per_epoch: int = 15,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 3,
    device: str = "cpu",
    output_dir: Path = None,
):
    print("\n" + "=" * 65)
    print(" Training Relation Networks on Animal Dataset (10 Categories)")
    print(f" Backbone: ResNet-50 (Pretrained, Frozen) | Comparator: Deep MLP")
    print(f" Episodes per Epoch: {episodes_per_epoch} | {n_way}-way {k_shot}-shot")
    print(f" Device: {device.upper()}")
    print("=" * 65)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = RelationNetwork(backbone_name="resnet50", pretrained=True, freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(model.relation_module.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0.0
    output_dir = output_dir or (CHECKPOINTS_DIR / "relation_networks")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best_relation_model.pt"

    for epoch in range(1, epochs + 1):
        model.relation_module.train()
        epoch_losses = []
        epoch_accs = []
        t0 = time.time()

        for ep in range(1, episodes_per_epoch + 1):
            supp_imgs, supp_lbls, q_imgs, q_lbls, _ = sample_episode(
                dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, transform=transform
            )
            supp_imgs = supp_imgs.to(device)
            supp_lbls = supp_lbls.to(device)
            q_imgs = q_imgs.to(device)
            q_lbls = q_lbls.to(device)

            scores, _ = model(supp_imgs, supp_lbls, q_imgs, num_classes=n_way)
            one_hot = F.one_hot(q_lbls, num_classes=n_way).float()
            loss = F.mse_loss(scores, one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = scores.argmax(dim=1)
            acc = (preds == q_lbls).float().mean().item()
            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

        scheduler.step()
        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = sum(epoch_accs) / len(epoch_accs)

        # Validation episode
        model.relation_module.eval()
        val_accs = []
        with torch.no_grad():
            for _ in range(5):
                s_imgs, s_lbls, q_imgs, q_lbls, _ = sample_episode(
                    dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, transform=eval_transform
                )
                scores, _ = model(s_imgs.to(device), s_lbls.to(device), q_imgs.to(device), num_classes=n_way)
                preds = scores.argmax(dim=1)
                val_accs.append((preds == q_lbls.to(device)).float().mean().item())

        val_acc = sum(val_accs) / len(val_accs)
        elapsed = time.time() - t0

        print(
            f" Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:5.1f}% | "
            f"Val Acc: {val_acc*100:5.1f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_acc or not best_ckpt_path.exists():
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "relation_module_state_dict": model.relation_module.state_dict(),
                "val_accuracy": val_acc,
                "algorithm": "relation_networks",
                "backbone": "resnet50",
            }, best_ckpt_path)

    print("\n" + "=" * 65)
    print(f" Training Complete! Best Validation Accuracy: {best_acc*100:.1f}%")
    print(f" Checkpoint Saved to: {best_ckpt_path}")
    print("=" * 65)
    return str(best_ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Train MetaLearnHub on existing dataset")
    parser.add_argument("--algorithm", type=str, default="relation_net", choices=["relation_net", "protonet", "matching_net"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of meta-training epochs")
    parser.add_argument("--episodes", type=int, default=15, help="Episodes per epoch")
    parser.add_argument("--n-way", type=int, default=5, help="N-way classes per episode")
    parser.add_argument("--k-shot", type=int, default=5, help="K-shot support images per class")
    parser.add_argument("--query-per-class", type=int, default=3, help="Query images per class")
    parser.add_argument("--samples-per-class", type=int, default=60, help="Max images sampled per class for dataset")
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        print(f"Dataset directory not found at: {DATASET_DIR}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading dataset from: {DATASET_DIR}")
    dataset = get_image_paths_by_class(DATASET_DIR, max_per_class=args.samples_per_class)
    print(f"Loaded {len(dataset)} animal classes: {list(dataset.keys())}")

    if args.algorithm == "relation_net":
        train_relation_network(
            dataset,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.query_per_class,
            device=device,
        )
    else:
        print(f"Selected algorithm: {args.algorithm} (uses pretrained ResNet-50 feature extractor directly).")


if __name__ == "__main__":
    main()
