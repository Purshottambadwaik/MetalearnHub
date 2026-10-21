"""
Meta-Learning Algorithms Package for MetaLearnHub.
"""

from ml.algorithms.base import MetaLearningAlgorithm
from ml.algorithms.prototypical_networks.protonet import (
    PrototypicalNetwork,
    PrototypicalNetworkAlgorithm,
)
from ml.algorithms.matching_networks.matching_net import (
    MatchingNetwork,
    MatchingNetworkAlgorithm,
)
from ml.algorithms.relation_networks.relation_net import (
    RelationNetwork,
    RelationNetworkAlgorithm,
)
from ml.algorithms.maml.maml import (
    MAML,
    MAMLAlgorithm,
)
from ml.algorithms.hybrid.ame_net import (
    AMENet,
    AMENetAlgorithm,
)


def get_algorithm(
    name: str = "prototypical_networks",
    backbone_name: str = "resnet50",
    pretrained: bool = True,
    device: str = "cpu",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = False,
    distance_metric: str = "cosine",
    **kwargs,
) -> MetaLearningAlgorithm:
    """
    Factory function to instantiate meta-learning algorithms.

    Supported names:
    - "prototypical_networks", "protonet"
    - "matching_networks", "matching_net"
    - "relation_networks", "relation_net"
    - "maml", "fomaml"
    - "ame_net", "amenet", "hybrid"
    """
    normalized_name = name.lower().replace("-", "_").strip()

    if normalized_name in ["prototypical_networks", "protonet"]:
        return PrototypicalNetworkAlgorithm(
            backbone_name=backbone_name,
            pretrained=pretrained,
            distance_metric=distance_metric,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            freeze_backbone=freeze_backbone,
        )
    elif normalized_name in ["matching_networks", "matching_net"]:
        return MatchingNetworkAlgorithm(
            backbone_name=backbone_name,
            pretrained=pretrained,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            freeze_backbone=freeze_backbone,
        )
    elif normalized_name in ["relation_networks", "relation_net"]:
        return RelationNetworkAlgorithm(
            backbone_name=backbone_name,
            pretrained=pretrained,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            freeze_backbone=freeze_backbone,
        )
    elif normalized_name in ["maml", "fomaml"]:
        return MAMLAlgorithm(
            backbone_name=backbone_name,
            pretrained=pretrained,
            learning_rate=learning_rate,
            device=device,
            freeze_backbone=freeze_backbone,
        )
    elif normalized_name in ["ame_net", "amenet", "hybrid"]:
        return AMENetAlgorithm(
            backbone_name=backbone_name,
            pretrained=pretrained,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            freeze_backbone=freeze_backbone,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown algorithm: '{name}'. Supported options are: "
            f"['prototypical_networks', 'matching_networks', 'relation_networks', 'maml', 'ame_net']"
        )
