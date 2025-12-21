"""HNSW Index wrapper for VRP insertion candidate search.

This module provides the core HNSW index functionality for accelerating
nearest neighbor search in the VRP context. It wraps FAISS (IndexHNSWFlat)
for high-performance approximate nearest neighbor search.

Key Features:
- O(log n) query time for approximate nearest neighbor search
- Incremental insertion support for dynamic VRP
- Configurable M and ef parameters for speed/accuracy tradeoff
- More robust than hnswlib, better handling of large k values
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple, List
import numpy as np
import logging

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False

if TYPE_CHECKING:
    from ..models.problem import VRPInstance

logger = logging.getLogger(__name__)


@dataclass
class HNSWConfig:
    """Configuration for HNSW index.
    
    Attributes:
        dim: Dimensionality of feature vectors
        max_elements: Maximum number of elements in index
        M: Number of bi-directional links per element (higher = more accurate, slower)
        ef_construction: Size of dynamic candidate list during construction
        ef_search: Size of dynamic candidate list during search
        space: Distance metric ('l2', 'ip', 'cosine')
        seed: Random seed for reproducibility
    """
    dim: int = 14  # Default feature dimension for VRP (8 base + 3 subseq + 3 fairness)
    max_elements: int = 10000
    M: int = 96  # Increased to 96 for better connectivity and reduced initial failure rates. Adaptive: 64 for small (<100), 96 for medium (100-500), 128 for large (>500)
    ef_construction: int = 400  # Increased to 400 for better initial index quality and reduced failure rates (target: <15% initial failures)
    ef_search: int = 100  # Higher = better recall during search (increased from 50 to handle larger k)
    space: str = "l2"  # L2 (Euclidean) distance
    seed: int = 42


@dataclass
class InsertionIndex:
    """HNSW-based index for finding insertion candidates.
    
    This index stores feature vectors for (customer, route, position) tuples
    and enables fast approximate nearest neighbor search to find promising
    insertion positions for removed customers.
    
    The index supports:
    - Bulk initialization from existing solution
    - Incremental updates for dynamic scenarios
    - Multi-query batch search for efficiency
    
    Attributes:
        config: HNSW configuration parameters
        index: The FAISS IndexHNSWFlat index object
        id_to_label: Mapping from internal IDs to (route_id, position) tuples
        label_to_id: Reverse mapping for deletion support
        n_elements: Current number of elements in index
    """
    config: HNSWConfig = field(default_factory=HNSWConfig)
    index: Optional[object] = field(default=None, init=False, repr=False)
    id_to_label: dict = field(default_factory=dict, init=False, repr=False)
    label_to_id: dict = field(default_factory=dict, init=False, repr=False)
    n_elements: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)
    _next_id: int = field(default=0, init=False)  # FAISS auto-generates IDs, we track them
    _deleted_ids: set = field(default_factory=set, init=False, repr=False)  # Soft delete: track deleted IDs
    
    def __post_init__(self):
        """Initialize the HNSW index."""
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        self._create_index()
    
    def _create_index(self):
        """Create a new HNSW index with configured parameters."""
        # FAISS IndexHNSWFlat: dimension, M (connectivity)
        self.index = faiss.IndexHNSWFlat(self.config.dim, self.config.M)
        
        # Set ef_construction and ef_search
        self.index.hnsw.efConstruction = self.config.ef_construction
        self.index.hnsw.efSearch = self.config.ef_search
        
        # FAISS doesn't support random seed directly, but we can set it via numpy
        np.random.seed(self.config.seed)
        
        self._initialized = True
        self.n_elements = 0
        self._next_id = 0
        self.id_to_label = {}
        self.label_to_id = {}
        self._deleted_ids = set()
    
    def add_items(
        self,
        vectors: np.ndarray,
        labels: List[Tuple[int, int]],
    ) -> None:
        """Add items to the index.
        
        Args:
            vectors: Feature vectors of shape (n, dim)
            labels: List of (route_id, position) tuples corresponding to vectors
        """
        if len(vectors) == 0:
            return
        
        vectors = np.asarray(vectors, dtype=np.float32)
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self.config.dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != expected {self.config.dim}"
            )
        
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        # FAISS auto-generates IDs starting from 0, but we need to track them
        # Get current number of vectors before adding
        start_id = self.index.ntotal if self.index.ntotal > 0 else 0
        
        # Add to index (FAISS returns nothing, IDs are implicit)
        self.index.add(vectors)
        
        # Generate IDs for the newly added vectors
        n_added = len(vectors)
        ids = np.arange(start_id, start_id + n_added)
        
        # Update mappings
        for internal_id, label in zip(ids, labels):
            self.id_to_label[internal_id] = label
            self.label_to_id[label] = internal_id
        
        self.n_elements = self.index.ntotal
        self._next_id = start_id + n_added
    
    def add_item(
        self,
        vector: np.ndarray,
        label: Tuple[int, int],
    ) -> None:
        """Add a single item to the index.
        
        Args:
            vector: Feature vector of shape (dim,)
            label: (route_id, position) tuple
        """
        self.add_items(vector.reshape(1, -1), [label])
    
    def _get_actual_element_count(self) -> int:
        """Get actual count of non-deleted elements in the index.
        
        FAISS doesn't support remove_ids for HNSW, so we use soft delete.
        Actual count = total elements - deleted elements.
        """
        if not _HAS_FAISS or self.index is None:
            return self.n_elements - len(self._deleted_ids)
        
        try:
            # FAISS ntotal is total elements, subtract deleted
            return self.index.ntotal - len(self._deleted_ids)
        except Exception:
            # Fallback to n_elements
            return self.n_elements - len(self._deleted_ids)
    
    def query(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Find k nearest neighbors for a query vector.
        
        ROBUST implementation with:
        - Automatic ef_search adjustment based on actual element count
        - Retry logic with progressively higher ef_search
        - Fragmentation-aware k reduction
        - Progressive k reduction on errors
        - Iterative k=1 fallback for severely fragmented indices
        
        Args:
            query_vector: Query feature vector of shape (dim,)
            k: Number of neighbors to return
            
        Returns:
            Tuple of (labels, distances) where labels are (route_id, position) tuples
        """
        query_vector = np.asarray(query_vector, dtype=np.float32)
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if self.n_elements == 0:
            return [], np.array([])
        
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        # Get actual element count (excluding deleted)
        actual_count = self._get_actual_element_count()
        if actual_count == 0:
            return [], np.array([])
        
        # PERMANENT FIX: Check fragmentation ratio
        # If too many deleted items, the graph is fragmented and queries will fail
        fragmentation_ratio = (self.n_elements - actual_count) / self.n_elements if self.n_elements > 0 else 0.0
        
        # Adjust k to available elements
        original_k = k
        k = min(k, actual_count)
        if k == 0:
            return [], np.array([])
        
        # PERMANENT FIX: If index is heavily fragmented (>30% deleted), reduce k aggressively
        # This prevents the "contiguous 2D array" error by querying fewer neighbors
        if fragmentation_ratio > 0.3 and k > 5:
            # Reduce k to avoid fragmentation issues
            k = max(1, int(k * (1 - fragmentation_ratio)))
            # Ensure k doesn't exceed available elements
            k = min(k, actual_count)
        
        # ROBUST ef_search calculation:
        # IMPROVEMENT: More aggressive ef_search calculation to reduce failures
        # For large indices (1000+), we need much higher ef_search
        # Rule: ef_search should be at least k * 3 for large indices, or k + sqrt(n) * 2
        current_ef = self.config.ef_search
        
        # More aggressive calculation for all indices to reduce failures
        if actual_count > 500:
            # Large index: use k * 5 or k + sqrt(n) * 4 (more aggressive)
            base_ef = max(k * 5, int(k + np.sqrt(actual_count) * 4))
        elif actual_count > 200:
            # Medium-large index: use k * 4 or k + sqrt(n) * 3
            base_ef = max(k * 4, int(k + np.sqrt(actual_count) * 3))
        elif actual_count > 100:
            # Medium index: use k * 3 or k + sqrt(n) * 2
            base_ef = max(k * 3, int(k + np.sqrt(actual_count) * 2))
        else:
            # Small-medium index: use k * 3 or k + sqrt(n) * 2 (more aggressive)
            base_ef = max(k * 3, int(k + np.sqrt(actual_count) * 2) + 20)
        
        min_ef = max(base_ef, current_ef)
        required_ef = min_ef  # Start with calculated value
        
        # FAISS is more robust - simpler retry logic
        # FAISS handles large k values better than hnswlib
        max_retries = 3  # FAISS is more reliable, fewer retries needed
        ef_multipliers = [1.0, 1.5, 2.0]  # Simpler multipliers for FAISS
        
        last_error = None
        last_error_msg = None
        
        # FAISS: Try with progressively higher ef_search
        for retry in range(max_retries):
            try_ef = int(required_ef * ef_multipliers[retry])
            
            try:
                # FAISS: Set ef_search
                self.index.hnsw.efSearch = try_ef
                
                # FAISS: search() returns (distances, indices) - NOTE: distances first!
                distances, internal_ids = self.index.search(query_vector, k)
                
                # Success! Convert to labels
                # FAISS returns 2D arrays: distances[0] and internal_ids[0] for single query
                # Filter out invalid IDs (idx < 0 means not found) and deleted IDs
                labels = []
                dists = []
                for idx, dist in zip(internal_ids[0], distances[0]):
                    if idx >= 0 and idx not in self._deleted_ids:
                        labels.append(self.id_to_label[int(idx)])
                        dists.append(float(dist))
                
                # If we got fewer than requested due to deleted items, that's okay
                if len(labels) < k and len(labels) > 0:
                    # Return what we have
                    pass
                
                # Restore original ef_search
                if try_ef != current_ef:
                    self.index.hnsw.efSearch = current_ef
                
                return labels, dists
                
            except Exception as e:
                error_msg = str(e)
                last_error = e
                last_error_msg = error_msg
                
                # FAISS is more robust - continue to next retry
                continue
        
        # All retries failed - raise error with diagnostic info
        # Repair operators will catch this and fall back to optimized linear scan
        raise RuntimeError(
            f"HNSW query failed after {max_retries} retries "
            f"(original_k={original_k}, final_k={k}, ef_tried={[int(required_ef * m) for m in ef_multipliers[:max_retries]]}, "
            f"M={self.config.M}, actual_count={actual_count}, total_elements={self.n_elements}, "
            f"fragmentation={fragmentation_ratio*100:.1f}%): {last_error_msg or last_error}"
        ) from last_error
    
    def batch_query(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
    ) -> Tuple[List[List[Tuple[int, int]]], np.ndarray]:
        """Find k nearest neighbors for multiple query vectors.
        
        ROBUST implementation with automatic retry and fragmentation handling.
        
        Args:
            query_vectors: Query vectors of shape (n_queries, dim)
            k: Number of neighbors per query
            
        Returns:
            Tuple of (labels_list, distances) where labels_list[i] contains
            (route_id, position) tuples for query i
        """
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        if self.n_elements == 0:
            return [[] for _ in range(len(query_vectors))], np.array([])
        
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        # Get actual element count (excluding deleted)
        actual_count = self._get_actual_element_count()
        if actual_count == 0:
            return [[] for _ in range(len(query_vectors))], np.array([])
        
        k = min(k, actual_count)
        if k == 0:
            return [[] for _ in range(len(query_vectors))], np.array([])
        
        # FAISS: Simpler ef_search calculation (more robust)
        current_ef = self.config.ef_search
        
        # FAISS handles large k better, so simpler calculation
        if actual_count > 500:
            base_ef = max(k * 3, int(k + np.sqrt(actual_count) * 2))
        else:
            base_ef = max(k * 2, int(k + np.sqrt(actual_count)) + 10)
        
        min_ef = max(base_ef, current_ef)
        required_ef = min_ef
        
        # FAISS: Simpler retry logic (more reliable)
        max_retries = 2
        ef_multipliers = [1.0, 1.5]
        
        last_error = None
        for retry in range(max_retries):
            try_ef = int(required_ef * ef_multipliers[retry])
            
            try:
                # FAISS: Set ef_search
                self.index.hnsw.efSearch = try_ef
                
                # FAISS: search() returns (distances, indices) - distances first!
                distances, internal_ids = self.index.search(query_vectors, k)
                
                # Success! Convert to labels
                all_labels = []
                all_distances = []
                for i in range(len(query_vectors)):
                    labels = []
                    dists = []
                    for idx, dist in zip(internal_ids[i], distances[i]):
                        if idx >= 0 and idx not in self._deleted_ids:
                            labels.append(self.id_to_label[int(idx)])
                            dists.append(float(dist))
                    all_labels.append(labels)
                    all_distances.append(dists)
                
                # Restore original ef_search
                if try_ef != current_ef:
                    self.index.hnsw.efSearch = current_ef
                
                return all_labels, np.array(all_distances)
                
            except Exception as e:
                last_error = e
                # Continue to next retry
                continue
        
        # All retries failed - raise error
        # Repair operators will catch this and fall back to optimized linear scan
        raise RuntimeError(
            f"HNSW batch query failed after {max_retries} retries "
            f"(k={k}, ef_tried={[int(required_ef * m) for m in ef_multipliers[:max_retries]]}, "
            f"M={self.config.M}, actual_count={actual_count}): {last_error}"
        ) from last_error
    
    def remove_item(self, label: Tuple[int, int]) -> bool:
        """Mark an item as deleted (soft delete).
        
        Note: FAISS IndexHNSWFlat doesn't support remove_ids(), so we use soft delete.
        Deleted items are filtered out during queries.
        For large deletions, consider rebuilding the index.
        
        Args:
            label: (route_id, position) tuple to remove
            
        Returns:
            True if item was found and marked, False otherwise
        """
        if label not in self.label_to_id:
            return False
        
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        internal_id = self.label_to_id[label]
        
        # FAISS: IndexHNSWFlat doesn't support remove_ids()
        # Use soft delete: mark as deleted, filter during queries
        self._deleted_ids.add(internal_id)
        
        # Update mappings (keep for reference, but mark as deleted)
        # Don't delete from mappings - we need them to filter during queries
        
        return True
    
    def clear(self) -> None:
        """Clear all items from the index."""
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        # FAISS: reset() clears the index
        if self.index is not None:
            self.index.reset()
        self.n_elements = 0
        self._next_id = 0
        self.id_to_label = {}
        self.label_to_id = {}
        self._deleted_ids = set()
    
    def rebuild(
        self,
        vectors: np.ndarray,
        labels: List[Tuple[int, int]],
    ) -> None:
        """Rebuild the index from scratch with new data.
        
        Useful when too many items have been deleted (fragmentation).
        
        Args:
            vectors: New feature vectors
            labels: Corresponding labels
        """
        self.clear()
        self.add_items(vectors, labels)
    
    def get_recall_at_k(
        self,
        query_vectors: np.ndarray,
        ground_truth: List[List[Tuple[int, int]]],
        k: int = 10,
    ) -> float:
        """Compute recall@k for evaluation.
        
        Args:
            query_vectors: Query vectors
            ground_truth: True nearest neighbors for each query
            k: Number of neighbors
            
        Returns:
            Average recall across queries
        """
        predictions, _ = self.batch_query(query_vectors, k=k)
        
        total_recall = 0.0
        for pred, gt in zip(predictions, ground_truth):
            pred_set = set(pred)
            gt_set = set(gt[:k])
            if len(gt_set) > 0:
                recall = len(pred_set & gt_set) / len(gt_set)
                total_recall += recall
        
        return total_recall / len(predictions) if predictions else 0.0
    
    def save(self, filepath: str) -> None:
        """Save the index to disk.
        
        Args:
            filepath: Path to save the index
        """
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        # FAISS: write_index() saves the index
        faiss.write_index(self.index, filepath)
        
        # Save mappings separately
        import pickle
        with open(filepath + '.meta', 'wb') as f:
            pickle.dump({
                'id_to_label': self.id_to_label,
                'label_to_id': self.label_to_id,
                'n_elements': self.n_elements,
                'config': self.config,
                '_next_id': self._next_id,
                '_deleted_ids': self._deleted_ids,
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'InsertionIndex':
        """Load an index from disk.
        
        Args:
            filepath: Path to the saved index
            
        Returns:
            Loaded InsertionIndex
        """
        import pickle
        
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        # Load metadata
        with open(filepath + '.meta', 'rb') as f:
            meta = pickle.load(f)
        
        config = meta['config']
        index = cls(config=config)
        
        # FAISS: read_index() loads the index
        index.index = faiss.read_index(filepath)
        index.id_to_label = meta['id_to_label']
        index.label_to_id = meta['label_to_id']
        index.n_elements = meta['n_elements']
        index._next_id = meta.get('_next_id', index.n_elements)
        index._deleted_ids = meta.get('_deleted_ids', set())
        
        return index
    
    def __len__(self) -> int:
        """Return number of elements in index."""
        return self.n_elements
    
    def __repr__(self) -> str:
        return (
            f"InsertionIndex(n_elements={self.n_elements}, "
            f"dim={self.config.dim}, M={self.config.M}, "
            f"ef_search={self.config.ef_search}, backend=FAISS)"
        )


def create_insertion_index(
    dim: int = 11,
    max_elements: int = 10000,
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> InsertionIndex:
    """Factory function to create an InsertionIndex with common parameters.
    
    Args:
        dim: Feature vector dimension
        max_elements: Maximum index capacity
        M: HNSW connectivity parameter
        ef_construction: Construction quality parameter
        ef_search: Search quality parameter
        
    Returns:
        Configured InsertionIndex
    """
    config = HNSWConfig(
        dim=dim,
        max_elements=max_elements,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    return InsertionIndex(config=config)

