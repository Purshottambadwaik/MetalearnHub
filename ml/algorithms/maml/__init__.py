"""
MAML (Model-Agnostic Meta-Learning) / First-Order MAML Package.
Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017.
"""

from .maml import MAML, MAMLAlgorithm

__all__ = ["MAML", "MAMLAlgorithm"]
