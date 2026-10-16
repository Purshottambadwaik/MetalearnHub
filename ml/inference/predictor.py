"""
Few-Shot Inference & Persistent Class-Incremental Registry for MetaLearnHub.

Handles the real-world application workflow:
1. Persistent Class Storage: Classes and prototypes are saved to disk (data/class_registry/).
2. Base & User Classes: Both pre-trained base classes and newly created user classes coexist.
3. Class Collision & Contribution: Detects existing class names (e.g., "Tiger") and allows
   contributing more support images to refine the prototype via running averages without retraining.
4. Open-Set Recognition: Distinguishes between images that match a registered class
   and images that are NOT in the system (out-of-distribution distance thresholding).
"""

import os
import time
import uuid
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from io import BytesIO
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from ml.config import PreprocessConfig, CLASS_REGISTRY_DIR, RAW_DATA_DIR, BREED_NAMES
from ml.datasets.cattle_dataset import get_eval_transforms
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm

logger = logging.getLogger(__name__)


class RegisteredClass:
    """
    Represents a registered class with persistent storage and a computed prototype.
    Supports both base dataset classes and dynamic user-created classes.
    """

    def __init__(
        self,
        class_name: str,
        class_id: str = None,
        class_type: str = "user",  # "base" (original dataset) or "user" (user contributed)
        description: str = "",
        registry_dir: Path = None,
    ):
        self.class_name = class_name
        self.class_id = class_id or str(uuid.uuid4())[:8]
        self.class_type = class_type
        self.description = description
        self.registry_dir = Path(registry_dir) if registry_dir else CLASS_REGISTRY_DIR
        
        self.support_image_paths: List[str] = []
        self.support_embeddings: Optional[torch.Tensor] = None
        self.prototype: Optional[torch.Tensor] = None
        self.intra_class_radius: float = 0.0
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()

    @property
    def k_shot(self) -> int:
        return len(self.support_image_paths)

    @property
    def is_ready(self) -> bool:
        return self.prototype is not None

    @property
    def class_dir(self) -> Path:
        return self.registry_dir / self.class_id

    @property
    def images_dir(self) -> Path:
        return self.class_dir / "images"

    def save(self):
        """Save class metadata, support embeddings, and prototype to disk."""
        self.class_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'class_type': self.class_type,
            'description': self.description,
            'k_shot': self.k_shot,
            'is_ready': self.is_ready,
            'intra_class_radius': float(self.intra_class_radius),
            'support_images': self.support_image_paths,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
        with open(self.class_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.prototype is not None:
            torch.save(self.prototype.cpu(), self.class_dir / "prototype.pt")

        if self.support_embeddings is not None:
            torch.save(self.support_embeddings.cpu(), self.class_dir / "embeddings.pt")

    @classmethod
    def load_from_dir(cls, class_dir: Path, registry_dir: Path = None) -> Optional['RegisteredClass']:
        """Load a registered class from its disk directory."""
        meta_file = class_dir / "meta.json"
        if not meta_file.exists():
            return None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            instance = cls(
                class_name=meta.get('class_name', class_dir.name),
                class_id=meta.get('class_id', class_dir.name),
                class_type=meta.get('class_type', 'user'),
                description=meta.get('description', ''),
                registry_dir=registry_dir or class_dir.parent,
            )
            instance.created_at = meta.get('created_at', '')
            instance.updated_at = meta.get('updated_at', '')
            instance.intra_class_radius = float(meta.get('intra_class_radius', 0.0))
            instance.support_image_paths = [
                str(p) for p in meta.get('support_images', []) if os.path.exists(p)
            ]

            proto_path = class_dir / "prototype.pt"
            if proto_path.exists():
                instance.prototype = torch.load(proto_path, map_location='cpu', weights_only=False)

            emb_path = class_dir / "embeddings.pt"
            if emb_path.exists():
                instance.support_embeddings = torch.load(emb_path, map_location='cpu', weights_only=False)

            return instance
        except Exception as e:
            logger.error(f"Failed to load class from {class_dir}: {e}")
            return None

    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'class_type': self.class_type,
            'description': self.description,
            'k_shot': self.k_shot,
            'is_ready': self.is_ready,
            'intra_class_radius': round(float(self.intra_class_radius), 4),
            'support_images': self.support_image_paths,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


# Maintain alias for backward compatibility
TemporaryClass = RegisteredClass


class FewShotPredictor:
    """
    Manages persistent class registry and few-shot / open-set predictions.
    
    Features:
    1. Persistent registry (loads/saves classes to disk).
    2. Support for both base dataset classes and user-defined classes.
    3. Duplicate detection & collaborative support image contribution.
    4. Distance thresholding for open-set prediction ("in system or not").
    """

    def __init__(
        self,
        algorithm: PrototypicalNetworkAlgorithm,
        preprocess_config: PreprocessConfig = None,
        registry_dir: Path = None,
    ):
        self.algorithm = algorithm
        self.transform = get_eval_transforms(preprocess_config or PreprocessConfig())
        self.registry_dir = Path(registry_dir) if registry_dir else CLASS_REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.classes: Dict[str, RegisteredClass] = {}
        # Alias for backward compatibility
        self.temporary_classes = self.classes
        self._relation_module = None

        self._load_all_classes_from_disk()

    def get_relation_module(self, device):
        """Lazily initialize or load trained RelationModule for comparator scoring."""
        if self._relation_module is None:
            from ml.algorithms.relation_networks.relation_net import RelationModule
            from ml.config import CHECKPOINTS_DIR
            emb_dim = getattr(self.algorithm.get_model(), "embedding_dim", 2048)
            rel_mod = RelationModule(embedding_dim=emb_dim).to(device)
            ckpt_path = CHECKPOINTS_DIR / "relation_networks" / "best_relation_model.pt"
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location=device)
                    if "relation_module_state_dict" in ckpt:
                        rel_mod.load_state_dict(ckpt["relation_module_state_dict"])
                        logger.info("Loaded trained RelationModule checkpoint successfully.")
                except Exception as e:
                    logger.warning(f"Could not load relation checkpoint: {e}")
            rel_mod.eval()
            self._relation_module = rel_mod
        return self._relation_module

    def _load_all_classes_from_disk(self):
        """Load all saved classes from the registry directory."""
        if not self.registry_dir.exists():
            return

        loaded_count = 0
        for class_dir in self.registry_dir.iterdir():
            if class_dir.is_dir() and (class_dir / "meta.json").exists():
                rc = RegisteredClass.load_from_dir(class_dir, self.registry_dir)
                if rc:
                    self.classes[rc.class_id] = rc
                    loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} registered classes from {self.registry_dir}")

    def get_class_by_name(self, class_name: str) -> Optional[RegisteredClass]:
        """Case-insensitive search for a class by name."""
        norm_name = class_name.strip().lower()
        for cls in self.classes.values():
            if cls.class_name.strip().lower() == norm_name:
                return cls
        return None

    def create_class(
        self,
        class_name: str,
        class_type: str = "user",
        description: str = "",
        force_new: bool = False,
    ) -> Tuple[RegisteredClass, bool]:
        """
        Create a new class or return existing class if duplicate.
        
        Returns:
            (RegisteredClass, is_new: bool)
        """
        existing = self.get_class_by_name(class_name)
        if existing and not force_new:
            logger.info(f"Class '{class_name}' already exists (id={existing.class_id}).")
            return existing, False

        new_class = RegisteredClass(
            class_name=class_name.strip(),
            class_type=class_type,
            description=description,
            registry_dir=self.registry_dir,
        )
        self.classes[new_class.class_id] = new_class
        new_class.save()
        logger.info(f"Created registered class: '{class_name}' (id={new_class.class_id}, type={class_type})")
        return new_class, True

    def add_support_image(
        self,
        class_id: str,
        image_path: str = None,
        image_bytes: bytes = None,
    ) -> RegisteredClass:
        """Add a single support image to a class."""
        if class_id not in self.classes:
            raise ValueError(f"Unknown class_id: {class_id}")

        reg_class = self.classes[class_id]
        reg_class.images_dir.mkdir(parents=True, exist_ok=True)

        if image_path and os.path.exists(image_path):
            # Copy into class images folder for persistence
            ext = Path(image_path).suffix or ".jpg"
            target_path = reg_class.images_dir / f"img_{uuid.uuid4().hex[:6]}{ext}"
            shutil.copy2(image_path, target_path)
            reg_class.support_image_paths.append(str(target_path))
        elif image_bytes:
            target_path = reg_class.images_dir / f"img_{uuid.uuid4().hex[:6]}.jpg"
            with open(target_path, "wb") as f:
                f.write(image_bytes)
            reg_class.support_image_paths.append(str(target_path))
        else:
            raise ValueError("Must provide either valid image_path or image_bytes")

        reg_class.updated_at = datetime.now().isoformat()
        reg_class.prototype = None  # Invalidate prototype until rebuilt
        reg_class.support_embeddings = None
        reg_class.save()
        return reg_class

    def add_support_images_batch(
        self,
        class_id: str,
        image_bytes_list: List[bytes],
    ) -> RegisteredClass:
        """Add multiple support images at once."""
        for b in image_bytes_list:
            self.add_support_image(class_id, image_bytes=b)
        return self.classes[class_id]

    def build_prototype(self, class_id: str) -> RegisteredClass:
        """
        Compute the prototype for a class from its support images.
        Also calculates intra-class radius to facilitate open-set thresholding.
        """
        if class_id not in self.classes:
            raise ValueError(f"Unknown class_id: {class_id}")

        reg_class = self.classes[class_id]
        if reg_class.k_shot == 0:
            raise ValueError(f"Class '{reg_class.class_name}' has no support images.")

        embeddings = []
        self.algorithm.get_model().eval()
        device = next(self.algorithm.get_model().parameters()).device

        with torch.no_grad():
            for img_path in reg_class.support_image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(device)
                    emb = self.algorithm.get_model().embed(img_tensor)
                    emb = F.normalize(emb, p=2, dim=1)
                    embeddings.append(emb.cpu())
                except Exception as e:
                    logger.warning(f"Failed to embed support image {img_path}: {e}")

        if not embeddings:
            raise ValueError(f"Failed to embed any support images for class '{reg_class.class_name}'")

        all_embeddings = torch.cat(embeddings, dim=0)
        # Compute normalized class prototype on unit sphere
        prototype = all_embeddings.mean(dim=0, keepdim=True)
        prototype = F.normalize(prototype, p=2, dim=1)

        # Calculate intra-class dispersion in cosine distance (1 - cos_sim)
        sims = (all_embeddings * prototype).sum(dim=1)
        dists = 1.0 - sims
        mean_dist = dists.mean().item()
        std_dist = dists.std().item() if len(dists) > 1 else 0.0
        max_dist = dists.max().item() if len(dists) > 0 else mean_dist
        # For small few-shot sets (K <= 10), sample variance underestimates real-world variance.
        # Anchor the radius floor using the system's baseline class distribution (~0.45 - 0.55).
        base_radii = [c.intra_class_radius for c in self.classes.values() if c.class_type == "base" and c.intra_class_radius > 0]
        mean_base = float(np.mean(base_radii)) if base_radii else 0.43
        system_floor = mean_base * 1.20 if reg_class.class_type == "user" else 0.35
        intra_radius = max(mean_dist + 1.25 * std_dist, max_dist * 1.05, system_floor)

        # Check for prototype collisions (identical images registered under different names)
        for other_cid, other_class in self.classes.items():
            if other_cid != class_id and other_class.is_ready and other_class.prototype is not None:
                collision_sim = torch.cosine_similarity(prototype, other_class.prototype).item()
                if collision_sim > 0.98:
                    logger.warning(
                        f"Prototype Collision: Class '{reg_class.class_name}' is visually identical to "
                        f"existing class '{other_class.class_name}' (cosine similarity {collision_sim:.4f})."
                    )

        reg_class.support_embeddings = all_embeddings
        reg_class.prototype = prototype
        reg_class.intra_class_radius = intra_radius
        reg_class.updated_at = datetime.now().isoformat()
        reg_class.save()

        logger.info(
            f"Built prototype for '{reg_class.class_name}' ({reg_class.k_shot}-shot). "
            f"Radius: {intra_radius:.2f}"
        )
        return reg_class

    def contribute_and_update(
        self,
        class_id: str,
        image_bytes_list: List[bytes],
    ) -> RegisteredClass:
        """
        Allows a user to add more images to an existing class and immediately updates
        the prototype vector (continuous refinement without weight retraining).
        """
        self.add_support_images_batch(class_id, image_bytes_list)
        return self.build_prototype(class_id)

    def seed_base_classes_from_dataset(
        self,
        raw_data_dir: Path = None,
        max_samples_per_breed: int = 8,
    ) -> List[RegisteredClass]:
        """
        Precomputes and registers base dataset breeds so the system is aware
        of the initial trained dataset alongside user-added classes.
        """
        data_dir = Path(raw_data_dir) if raw_data_dir else RAW_DATA_DIR
        seeded_classes = []

        if not data_dir.exists():
            logger.warning(f"Cannot seed base classes; directory does not exist: {data_dir}")
            return seeded_classes

        for breed_dir in data_dir.iterdir():
            if not breed_dir.is_dir():
                continue

            breed_name = breed_dir.name
            existing = self.get_class_by_name(breed_name)
            if existing and existing.is_ready:
                seeded_classes.append(existing)
                continue

            # Create base class
            reg_class, _ = self.create_class(
                class_name=breed_name,
                class_type="base",
                description="Original base dataset breed",
                force_new=False,
            )

            # Collect base images (exclude _c duplicates if present to keep support clean)
            img_paths = [
                p for p in breed_dir.glob("*.jpg")
                if not p.stem.endswith("_c")
            ][:max_samples_per_breed]

            if not img_paths:
                img_paths = list(breed_dir.glob("*.jpg"))[:max_samples_per_breed]

            for p in img_paths:
                try:
                    self.add_support_image(reg_class.class_id, image_path=str(p))
                except Exception as e:
                    logger.warning(f"Error adding base image {p}: {e}")

            if reg_class.k_shot > 0:
                self.build_prototype(reg_class.class_id)
                seeded_classes.append(reg_class)

        logger.info(f"Seeded {len(seeded_classes)} base dataset classes into registry.")
        return seeded_classes

    def predict(
        self,
        query_image_path: str = None,
        query_image_bytes: bytes = None,
        class_ids: List[str] = None,
        distance_threshold: Optional[float] = None,
        algorithm: str = "protonet",
    ) -> dict:
        """
        Classifies a query image against registered classes with Open-Set Out-of-Distribution rejection.
        
        Supports selectable algorithms:
        - "protonet": Prototypical Networks (class centroids)
        - "matching_net": Matching Networks (exemplar attention)
        - "relation_net": Relation Networks (deep neural comparator)
        - "ensemble": Multi-algorithm weighted voting & distance fusion
        """
        # Load query image
        if query_image_path:
            query_img = Image.open(query_image_path).convert('RGB')
        elif query_image_bytes:
            query_img = Image.open(BytesIO(query_image_bytes)).convert('RGB')
        else:
            raise ValueError("Must provide either query_image_path or query_image_bytes")

        # By default, compare against ALL ready classes in the system (both base and user classes)
        if class_ids is None or len(class_ids) == 0:
            target_classes = [c for c in self.classes.values() if c.is_ready]
        else:
            target_classes = [self.classes[cid] for cid in class_ids if cid in self.classes and self.classes[cid].is_ready]

        if not target_classes:
            raise ValueError("No ready classes available with computed prototypes.")

        # Stack prototypes
        prototypes = [tc.prototype for tc in target_classes]
        all_prototypes = torch.cat(prototypes, dim=0)
        all_prototypes = F.normalize(all_prototypes, p=2, dim=1)

        # Device setup
        self.algorithm.get_model().eval()
        device = next(self.algorithm.get_model().parameters()).device
        all_prototypes = all_prototypes.to(device)

        query_tensor = self.transform(query_img).unsqueeze(0).to(device)
        algo_mode = algorithm.lower().replace("-", "_").strip() if algorithm else "protonet"

        t_start = time.time()
        with torch.no_grad():
            query_embedding = self.algorithm.get_model().embed(query_tensor)
            query_embedding = F.normalize(query_embedding, p=2, dim=1)

            # 1. Prototypical Networks (ProtoNet) computation:
            proto_sims = torch.mm(query_embedding, all_prototypes.t())
            proto_dists = 1.0 - proto_sims
            proto_probs = torch.softmax(proto_sims * 16.0, dim=1)[0]

            # 2. Matching Networks (MatchingNet) exemplar-attention computation:
            all_supps = []
            supp_labels = []
            for c_idx, tc in enumerate(target_classes):
                s_emb = tc.support_embeddings if (tc.support_embeddings is not None and len(tc.support_embeddings) > 0) else tc.prototype
                s_emb = F.normalize(s_emb.to(device), p=2, dim=1)
                all_supps.append(s_emb)
                supp_labels.extend([c_idx] * s_emb.size(0))
            stacked_supps = torch.cat(all_supps, dim=0)
            s_labels = torch.tensor(supp_labels, device=device, dtype=torch.long)
            s_sims = torch.mm(query_embedding, stacked_supps.t())
            s_dists = 1.0 - s_sims
            s_attn = F.softmax(s_sims * 16.0, dim=1)
            one_hot = F.one_hot(s_labels, num_classes=len(target_classes)).float()
            raw_class_attn = torch.mm(s_attn, one_hot)[0]
            # Normalize by shot count per class so 30-shot base classes don't drown out 5-shot user classes
            shot_counts = torch.tensor([max(1, tc.k_shot) for tc in target_classes], device=device, dtype=torch.float)
            balanced_attn = raw_class_attn / shot_counts
            matching_probs = balanced_attn / balanced_attn.sum()
            matching_dists = torch.zeros((1, len(target_classes)), device=device)
            for c_idx in range(len(target_classes)):
                mask = (s_labels == c_idx)
                matching_dists[0, c_idx] = s_dists[0, mask].min()

            # 3. Relation Networks (RelationNet) non-linear scoring:
            rel_mod = self.get_relation_module(device)
            q_rep = query_embedding.repeat(len(target_classes), 1)
            relation_scores = rel_mod(q_rep, all_prototypes)
            relation_dists = (1.0 - relation_scores).unsqueeze(0)
            relation_probs = F.softmax(relation_scores * 16.0, dim=0)

            # 4. MAML (Model-Agnostic Meta-Learning) inner-loop adaptation:
            maml_probs = None
            maml_dists = None
            if algo_mode in ["maml", "fomaml", "ensemble", "all", "fusion"]:
                from ml.algorithms.maml.maml import MAML
                maml_model = MAML(backbone_name="resnet50", pretrained=False)
                maml_head = maml_model.adapt_head(
                    stacked_supps, s_labels, num_classes=len(target_classes), steps=5, lr=0.05
                )
                with torch.no_grad():
                    maml_w_norm = F.normalize(maml_head.weight, p=2, dim=1)
                    maml_sims = torch.mm(query_embedding, maml_w_norm.t())
                    maml_dists = 1.0 - maml_sims
                    maml_probs = F.softmax(maml_sims * 16.0, dim=1)[0]

            # Route to chosen algorithm or ensemble:
            if algo_mode in ["matching_net", "matching_networks", "matching"]:
                active_algorithm_name = "Matching Networks"
                distances = matching_dists
                probs = matching_probs
            elif algo_mode in ["relation_net", "relation_networks", "relation"]:
                active_algorithm_name = "Relation Networks"
                distances = relation_dists
                probs = relation_probs
            elif algo_mode in ["maml", "fomaml"]:
                active_algorithm_name = "MAML (Model-Agnostic Meta-Learning)"
                distances = maml_dists
                probs = maml_probs
            elif algo_mode in ["ensemble", "all", "fusion"]:
                active_algorithm_name = "Ensemble (ProtoNet + MatchingNet + RelationNet + MAML)"
                probs = 0.35 * proto_probs + 0.30 * matching_probs + 0.20 * relation_probs + 0.15 * maml_probs
                distances = 0.35 * proto_dists + 0.30 * matching_dists + 0.20 * relation_dists + 0.15 * maml_dists
            else:
                active_algorithm_name = "Prototypical Networks"
                distances = proto_dists
                probs = proto_probs

            algorithm_breakdown = {
                "protonet": {
                    "class": target_classes[proto_probs.argmax().item()].class_name,
                    "confidence": round(float(proto_probs.max().item()), 4),
                    "distance": round(float(proto_dists.min().item()), 4),
                },
                "matching_net": {
                    "class": target_classes[matching_probs.argmax().item()].class_name,
                    "confidence": round(float(matching_probs.max().item()), 4),
                    "distance": round(float(matching_dists.min().item()), 4),
                },
                "relation_net": {
                    "class": target_classes[relation_probs.argmax().item()].class_name,
                    "confidence": round(float(relation_probs.max().item()), 4),
                    "distance": round(float(relation_dists.min().item()), 4),
                },
            }
            if maml_probs is not None:
                algorithm_breakdown["maml"] = {
                    "class": target_classes[maml_probs.argmax().item()].class_name,
                    "confidence": round(float(maml_probs.max().item()), 4),
                    "distance": round(float(maml_dists.min().item()), 4),
                }
        inference_time = time.time() - t_start

        # Find best class candidate (highest probability)
        pred_idx = torch.argmax(probs).item()
        min_distance = float(distances[0, pred_idx].item())
        confidence = float(probs[pred_idx].item())
        candidate_class = target_classes[pred_idx]

        # =================================================================
        # ADAPTIVE Open-Set Rejection -- Strict Distance & Confidence System
        # =================================================================
        # Gate 1 - Distance: query must fit within candidate's intra-class radius.
        # Gate 2 - Margin: clear winner over competitors.
        # Gate 3 - Confidence: probability must beat 2x random baseline.
        # =================================================================
        if distance_threshold is not None and distance_threshold > 0:
            threshold = float(distance_threshold)
        else:
            cand_radius = candidate_class.intra_class_radius
            if cand_radius > 0:
                threshold = float(cand_radius * 1.0)
            else:
                valid_radii = [c.intra_class_radius for c in target_classes if c.intra_class_radius > 0]
                threshold = float(np.mean(valid_radii)) if valid_radii else 0.22

        # Gate 2: clear margin over nearest competitor
        if len(target_classes) > 1:
            all_dists_sorted = sorted([float(distances[0, i].item()) for i in range(len(target_classes))])
            second_best_dist = all_dists_sorted[1]
            relative_gap = second_best_dist - min_distance
            has_margin = (relative_gap >= 0.005) or (min_distance <= cand_radius * 0.95)
        else:
            has_margin = True
            relative_gap = 1.0

        # Gate 1: distance within class radius OR high-confidence dominant match
        is_dominant_match = (confidence >= 0.85 and relative_gap >= 0.10 and min_distance <= 0.60)
        has_low_distance = (min_distance <= threshold) or is_dominant_match

        # Gate 3: confidence must be decisive (>= 50% for confirmed match).
        # In a multi-class system, any candidate with < 50% probability is heavily ambiguous / noisy.
        n_classes = len(target_classes)
        confidence_gate = 0.50
        has_sufficient_confidence = (confidence >= confidence_gate)

        is_in_system = bool(has_low_distance and has_margin and has_sufficient_confidence)
        status = "MATCHED" if is_in_system else "NOT_IN_SYSTEM"

        if is_in_system:
            verdict_text = candidate_class.class_name
            status_message = (
                f"Matched with '{candidate_class.class_name}' ({candidate_class.class_type.capitalize()} Class) "
                f"at distance {min_distance:.3f} (radius: {threshold:.3f}, confidence {confidence*100:.1f}%)."
            )
        else:
            verdict_text = "Unknown / Match Not Found"
            reasons = []
            if not has_low_distance:
                reasons.append(f"distance {min_distance:.3f} > allowable radius {threshold:.3f}")
            if not has_margin:
                reasons.append(f"competing too closely with other classes")
            if not has_sufficient_confidence:
                reasons.append(f"confidence {confidence*100:.1f}% < minimum certainty threshold {confidence_gate*100:.1f}%")
            status_message = (
                f"MATCH NOT FOUND: Image does not match any registered class in the system. "
                f"({' | '.join(reasons)})"
            )

        # Build detailed class similarities
        all_classes_result = []
        for i, tc in enumerate(target_classes):
            all_classes_result.append({
                'class_name': tc.class_name,
                'class_id': tc.class_id,
                'class_type': tc.class_type,
                'probability': round(float(probs[i].item()), 4),
                'distance': round(float(distances[0, i].item()), 4),
                'k_shot': tc.k_shot,
                'intra_class_radius': round(float(tc.intra_class_radius), 2),
            })

        # Sort by distance ascending
        all_classes_result.sort(key=lambda x: x['distance'])

        result = {
            'is_in_system': is_in_system,
            'status': status,
            'algorithm': active_algorithm_name,
            'algorithm_key': algo_mode,
            'algorithm_breakdown': algorithm_breakdown,
            'predicted_class': verdict_text,
            'predicted_class_id': candidate_class.class_id if is_in_system else None,
            'predicted_class_type': candidate_class.class_type if is_in_system else "unknown",
            'confidence': round(confidence, 4) if is_in_system else 0.0,
            'min_distance': round(min_distance, 4),
            'distance_threshold': round(threshold, 4),
            'status_message': status_message,
            'inference_time_ms': round(inference_time * 1000, 2),
            'total_classes_evaluated': len(target_classes),
            'all_classes': all_classes_result,
        }

        return result

    def list_classes(self) -> List[dict]:
        """List all registered classes in the system."""
        return [tc.to_dict() for tc in self.classes.values()]

    def remove_class(self, class_id: str) -> bool:
        """Remove a class from registry and disk."""
        if class_id in self.classes:
            target_class = self.classes.pop(class_id)
            if target_class.class_dir.exists():
                try:
                    shutil.rmtree(target_class.class_dir)
                except Exception as e:
                    logger.error(f"Error removing class dir {target_class.class_dir}: {e}")
            return True
        return False
