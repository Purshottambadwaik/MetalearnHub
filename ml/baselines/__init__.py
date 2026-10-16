"""
Conventional Machine Learning Baselines for MetaLearnHub.
Used for scientific benchmarking against Few-Shot Meta-Learning algorithms.
"""

from .standard_classifier import ConventionalClassifierBaseline, run_comparative_benchmark

__all__ = ["ConventionalClassifierBaseline", "run_comparative_benchmark"]
