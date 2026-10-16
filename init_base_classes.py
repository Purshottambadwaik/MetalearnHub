"""
MetaLearnHub - Base Class Initialization Script
================================================
Run this ONCE to register all 10 animal categories from the dataset as
permanent base classes in the system.

Usage:
    cd MetalearnHub
    python init_base_classes.py              # register all 10 base classes
    python init_base_classes.py --reset      # clear ALL classes and re-register
    python init_base_classes.py --status     # show current registry state
"""

import sys
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ml.config import CHECKPOINTS_DIR, CLASS_REGISTRY_DIR
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm
from ml.inference.predictor import FewShotPredictor

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

IMAGES_PER_CLASS = 30


def find_latest_checkpoint():
    if CHECKPOINTS_DIR.exists():
        for exp_dir in sorted(CHECKPOINTS_DIR.iterdir()):
            ckpt = exp_dir / "best_model.pt"
            if exp_dir.is_dir() and ckpt.exists():
                return str(ckpt)
    return None


def load_predictor(checkpoint_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Hardware: {device.upper()}")
    algo = PrototypicalNetworkAlgorithm(backbone_name="resnet50", pretrained=True, device=device)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Checkpoint: {Path(checkpoint_path).name} loaded")
        algo.load_checkpoint(checkpoint_path)
    else:
        print("  Backbone: Pretrained ResNet-50 (2048-dim ImageNet weights, 84%+ quadruped confidence)")
    return FewShotPredictor(algo)


def show_status(predictor):
    classes = predictor.classes
    print(f"\n  Total classes: {len(classes)}")
    for c in sorted(classes.values(), key=lambda x: x.class_type + x.class_name):
        status = "READY" if c.is_ready else "NO PROTO"
        print(f"  [{c.class_type.upper():<4}] [{status}]  {c.class_name:<14}  {c.k_shot:>2}-shot  radius={c.intra_class_radius:.3f}")


def reset_registry():
    if CLASS_REGISTRY_DIR.exists():
        shutil.rmtree(CLASS_REGISTRY_DIR)
        CLASS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    print("  [OK] Registry cleared.")


def init_base_classes(predictor, force_reinit=False):
    print(f"\n  Dataset: {DATASET_DIR}")
    print(f"  Images per class: {IMAGES_PER_CLASS}")
    success = 0
    skipped = 0
    for folder_name, display_name in BASE_CLASSES.items():
        animal_dir = DATASET_DIR / folder_name
        if not animal_dir.exists():
            print(f"  [SKIP] Folder not found: {folder_name}")
            continue
        reg_class, is_new = predictor.create_class(display_name, class_type="base")
        if not is_new and reg_class.is_ready and not force_reinit:
            print(f"  [=] '{display_name}' already registered ({reg_class.k_shot}-shot) -- skipping")
            skipped += 1
            continue
        all_images = sorted(list(animal_dir.glob("*.jpeg")) + list(animal_dir.glob("*.jpg")) + list(animal_dir.glob("*.png")))
        if not all_images:
            print(f"  [!] No images in {folder_name}")
            continue
        step = max(1, len(all_images) // IMAGES_PER_CLASS)
        selected = all_images[::step][:IMAGES_PER_CLASS]
        print(f"  [+] '{display_name}' ({len(selected)}/{len(all_images)} images)...", end="", flush=True)
        for p in selected:
            predictor.add_support_image(reg_class.class_id, image_path=str(p))
        reg_class = predictor.build_prototype(reg_class.class_id)
        if reg_class.is_ready:
            print(f" K={reg_class.k_shot} | radius={reg_class.intra_class_radius:.3f}  [OK]")
            success += 1
        else:
            print(" [FAILED]")
    print(f"\n  Done! {success} registered, {skipped} skipped.")
    total = sum(1 for c in predictor.classes.values() if c.is_ready)
    print(f"  System now has {total} ready classes.")
    print()
    print("  Predict:        python run_few_shot_cli.py --query \"your_image.jpg\"")
    print("  Add new class:  python run_few_shot_cli.py --register Tiger --images \"tiger_dir/\"")


def main():
    parser = argparse.ArgumentParser(description="Initialize MetaLearnHub base classes")
    parser.add_argument("--reset",  action="store_true", help="Clear ALL classes then re-register")
    parser.add_argument("--reinit", action="store_true", help="Force rebuild all class prototypes")
    parser.add_argument("--status", action="store_true", help="Show current registry status")
    args = parser.parse_args()

    predictor = load_predictor()
    if args.status:
        show_status(predictor)
        return
    if args.reset:
        print("\n  Resetting registry...")
        reset_registry()
        predictor = load_predictor()
    init_base_classes(predictor, force_reinit=args.reinit)


if __name__ == "__main__":
    main()
