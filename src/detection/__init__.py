"""
src/detection module initialization.

Detection architecture and data interfaces for spatio-temporal graph-based
attack detection in Water Distribution Systems (WDS).
"""

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.preprocessing import DataPreprocessor
from src.detection.features import FeatureExtractor
from src.detection.graph_model import SpatioTemporalGNN
from src.detection.train import train_model
from src.detection.evaluate import DetectionEvaluator

__all__ = [
    "BATADALDatasetLoader",
    "DataPreprocessor",
    "FeatureExtractor",
    "SpatioTemporalGNN",
    "train_model",
    "DetectionEvaluator",
]
