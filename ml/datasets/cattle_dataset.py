"""
PyTorch Dataset and Episodic Sampler for MetaLearnHub.

Provides torch Dataset classes and episodic batch samplers
for meta-learning training and evaluation.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image

from ml.config import PreprocessConfig, BREED_NAMES


# ============================================================
# Image Transforms
# ============================================================

def get_train_transforms(config: PreprocessConfig) -> transforms.Compose:
    """Get training transforms with augmentation."""
    transform_list = [
        transforms.Resize((config.image_size, config.image_size)),
    ]
    
    if config.use_augmentation:
        if config.random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if config.random_rotation > 0:
            transform_list.append(
                transforms.RandomRotation(config.random_rotation)
            )
        transform_list.append(
            transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
            )
        )
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ])
    
    return transforms.Compose(transform_list)


def get_eval_transforms(config: PreprocessConfig) -> transforms.Compose:
    """Get evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ])


# ============================================================
# MetaLearn Dataset
# ============================================================

class CattleDataset(Dataset):
    """
    PyTorch Dataset for cattle breed images.
    
    Stores images organized by class for efficient episodic sampling.
    """
    
    def __init__(
        self,
        image_paths: List[Tuple[str, str]],  # (filepath, breed_name)
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            image_paths: List of (filepath, breed_name) tuples
            transform: Image transform pipeline
            class_to_idx: Optional class-to-index mapping
        """
        self.image_paths = image_paths
        self.transform = transform
        
        # Build class-to-index mapping
        classes = sorted(set(breed for _, breed in image_paths))
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Build index of images per class
        self.class_images: Dict[str, List[int]] = defaultdict(list)
        for idx, (_, breed) in enumerate(image_paths):
            self.class_images[breed].append(idx)
        
        self.classes = classes
        self.num_classes = len(classes)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, breed = self.image_paths[idx]
        label = self.class_to_idx[breed]
        
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_images(self, class_name: str) -> List[int]:
        """Get indices of all images for a given class."""
        return self.class_images.get(class_name, [])


# ============================================================
# Episodic Sampler
# ============================================================

class EpisodicSampler:
    """
    Samples N-way K-shot episodes for meta-learning.
    
    Each episode consists of:
    - N classes sampled from available classes
    - K support images per class
    - Q query images per class
    
    Support and query sets are disjoint within each episode.
    """
    
    def __init__(
        self,
        dataset: CattleDataset,
        n_way: int,
        k_shot: int,
        query_per_class: int,
        num_episodes: int,
        seed: int = 42,
    ):
        """
        Args:
            dataset: The cattle dataset
            n_way: Number of classes per episode
            k_shot: Support images per class
            query_per_class: Query images per class
            num_episodes: Number of episodes to generate
            seed: Random seed
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.num_episodes = num_episodes
        self.rng = random.Random(seed)
        
        # Validate
        if n_way > dataset.num_classes:
            raise ValueError(
                f"n_way ({n_way}) > available classes ({dataset.num_classes}). "
                f"Available: {dataset.classes}"
            )
        
        # Check each class has enough images
        total_needed = k_shot + query_per_class
        for cls in dataset.classes:
            n_images = len(dataset.get_class_images(cls))
            if n_images < total_needed:
                raise ValueError(
                    f"Class '{cls}' has only {n_images} images, "
                    f"need at least {total_needed} (k_shot={k_shot} + "
                    f"query={query_per_class})"
                )
    
    def __len__(self) -> int:
        return self.num_episodes
    
    def __iter__(self):
        """
        Yields episodes as dictionaries:
        {
            'support_images': Tensor [N*K, C, H, W],
            'support_labels': Tensor [N*K],
            'query_images': Tensor [N*Q, C, H, W],
            'query_labels': Tensor [N*Q],
            'class_names': List[str],  # the N classes in this episode
        }
        """
        for _ in range(self.num_episodes):
            yield self._sample_episode()
    
    def _sample_episode(self) -> dict:
        """Sample a single episode."""
        # Select N random classes
        episode_classes = self.rng.sample(self.dataset.classes, self.n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for local_label, cls_name in enumerate(episode_classes):
            # Get all image indices for this class
            all_indices = list(self.dataset.get_class_images(cls_name))
            self.rng.shuffle(all_indices)
            
            # Split into support and query (disjoint)
            support_indices = all_indices[:self.k_shot]
            query_indices = all_indices[self.k_shot:self.k_shot + self.query_per_class]
            
            # Verify no overlap
            assert len(set(support_indices) & set(query_indices)) == 0, \
                "Support and query sets overlap!"
            
            # Load images
            for idx in support_indices:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(local_label)
            
            for idx in query_indices:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(local_label)
        
        return {
            'support_images': torch.stack(support_images),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'class_names': episode_classes,
        }


# ============================================================
# Few-Shot Evaluation Dataset
# ============================================================

class FewShotEvalDataset:
    """
    Creates few-shot evaluation episodes for the held-out class.
    
    For each evaluation:
    1. Sample K support images from the unseen class
    2. Use remaining unseen images as queries
    3. Optionally include some known classes for N-way comparison
    """
    
    def __init__(
        self,
        unseen_images: List[Tuple[str, str]],  # (path, breed) for held-out class
        known_images: List[Tuple[str, str]],    # (path, breed) for meta-train classes
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.unseen_images = unseen_images
        self.known_images = known_images
        self.transform = transform
        self.rng = random.Random(seed)
        
        # Organize known images by class
        self.known_by_class: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for path, breed in known_images:
            self.known_by_class[breed].append((path, breed))
    
    def sample_episode(
        self,
        k_shot: int,
        n_way: int = 2,
        query_per_class: int = 5,
        include_known_classes: bool = True,
    ) -> dict:
        """
        Sample a few-shot evaluation episode.
        
        The unseen class is always one of the N ways.
        Other N-1 ways are sampled from known classes.
        
        Args:
            k_shot: Number of support images for the unseen class
            n_way: Total number of classes in the episode
            query_per_class: Query images per class
            include_known_classes: Whether to include known classes for comparison
        
        Returns:
            Episode dict with support/query data
        """
        unseen_shuffled = list(self.unseen_images)
        self.rng.shuffle(unseen_shuffled)
        
        unseen_breed = unseen_shuffled[0][1]
        
        # Support from unseen class
        support_unseen = unseen_shuffled[:k_shot]
        # Query from unseen class (disjoint from support)
        remaining_unseen = unseen_shuffled[k_shot:]
        query_unseen = remaining_unseen[:query_per_class]
        
        if len(support_unseen) < k_shot:
            raise ValueError(
                f"Not enough unseen images for {k_shot}-shot. "
                f"Have {len(self.unseen_images)}, need at least {k_shot + query_per_class}"
            )
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        class_names = [unseen_breed]  # Unseen class = label 0
        
        # Load unseen support
        for path, _ in support_unseen:
            img = self._load_image(path)
            support_images.append(img)
            support_labels.append(0)
        
        # Load unseen query
        for path, _ in query_unseen:
            img = self._load_image(path)
            query_images.append(img)
            query_labels.append(0)
        
        # Add known classes
        if include_known_classes and n_way > 1:
            available_known = list(self.known_by_class.keys())
            n_known = min(n_way - 1, len(available_known))
            selected_known = self.rng.sample(available_known, n_known)
            
            for local_label, cls_name in enumerate(selected_known, start=1):
                class_names.append(cls_name)
                cls_images = list(self.known_by_class[cls_name])
                self.rng.shuffle(cls_images)
                
                # Support from known class
                for path, _ in cls_images[:k_shot]:
                    img = self._load_image(path)
                    support_images.append(img)
                    support_labels.append(local_label)
                
                # Query from known class
                for path, _ in cls_images[k_shot:k_shot + query_per_class]:
                    img = self._load_image(path)
                    query_images.append(img)
                    query_labels.append(local_label)
        
        return {
            'support_images': torch.stack(support_images),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'class_names': class_names,
            'unseen_class': unseen_breed,
            'k_shot': k_shot,
            'n_way': len(class_names),
        }
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and transform a single image."""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
