"""HNSW index module for fast insertion position lookup."""

from .index import InsertionIndex, HNSWConfig, create_insertion_index, _HAS_FAISS
from .features import FeatureEncoder, FeatureConfig, create_feature_encoder
from .manager import HNSWManager, HNSWManagerConfig, InsertionCandidate, create_hnsw_manager

__all__ = [
    # Index
    "InsertionIndex",
    "HNSWConfig",
    "create_insertion_index",
    "_HAS_FAISS",
    # Features
    "FeatureEncoder",
    "FeatureConfig",
    "create_feature_encoder",
    # Manager
    "HNSWManager",
    "HNSWManagerConfig",
    "InsertionCandidate",
    "create_hnsw_manager",
]
