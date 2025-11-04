"""CDPQ utilities (features, distillation, confidence heuristics)."""

from .features import FeatureCapture, capture_features
from .distillation import (
    DistillationResult,
    compute_distillation_loss,
    compute_feature_affinity_loss,
    compute_layer_statistics_loss,
)
from .confidence import confidence_baseline, confidence_variance

__all__ = [
    "FeatureCapture",
    "capture_features",
    "DistillationResult",
    "compute_distillation_loss",
    "compute_feature_affinity_loss",
    "compute_layer_statistics_loss",
    "confidence_baseline",
    "confidence_variance",
]
