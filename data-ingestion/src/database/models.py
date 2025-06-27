"""
Database Models

This module defines configuration and result classes for vector database operations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    COSINE = "Cosine"
    DOT = "Dot"
    EUCLIDEAN = "Euclidean"
    MANHATTAN = "Manhattan"


@dataclass
class VectorConfig:
    """Configuration for vector database operations."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    https: bool = False
    api_key: Optional[str] = None
    
    # Collection settings
    collection_name: str = "tekyz_knowledge"
    vector_size: int = 384  # For all-MiniLM-L6-v2
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # Performance settings
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 60.0
    
    # Indexing settings
    hnsw_config: Optional[Dict[str, Any]] = None
    optimizers_config: Optional[Dict[str, Any]] = None
    wal_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default configurations after initialization."""
        if self.hnsw_config is None:
            self.hnsw_config = {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 0
            }
        
        if self.optimizers_config is None:
            self.optimizers_config = {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 0,
                "max_segment_size": None,
                "memmap_threshold": None,
                "indexing_threshold": 20000,
                "flush_interval_sec": 5,
                "max_optimization_threads": 1
            }


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None
    
    @property
    def chunk_id(self) -> str:
        """Get chunk ID from payload."""
        return self.payload.get('chunk_id', self.id)
    
    @property
    def text(self) -> str:
        """Get text content from payload."""
        return self.payload.get('text', '')
    
    @property
    def clean_text(self) -> str:
        """Get clean text content from payload."""
        return self.payload.get('clean_text', self.text)
    
    @property
    def source_url(self) -> str:
        """Get source URL from payload."""
        return self.payload.get('source_url', '')
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata from payload."""
        return self.payload.get('metadata', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'score': self.score,
            'chunk_id': self.chunk_id,
            'text': self.text,
            'clean_text': self.clean_text,
            'source_url': self.source_url,
            'metadata': self.metadata,
            'payload': self.payload
        }


@dataclass
class UploadResult:
    """Result from vector upload operation."""
    
    success: bool
    total_vectors: int
    uploaded_vectors: int
    failed_vectors: int
    upload_time: float
    errors: List[str]
    operation_id: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_vectors == 0:
            return 0.0
        return self.uploaded_vectors / self.total_vectors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'total_vectors': self.total_vectors,
            'uploaded_vectors': self.uploaded_vectors,
            'failed_vectors': self.failed_vectors,
            'upload_time': self.upload_time,
            'success_rate': self.success_rate,
            'errors': self.errors,
            'operation_id': self.operation_id
        }


@dataclass
class CollectionInfo:
    """Information about a vector collection."""
    
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    config: Dict[str, Any]
    payload_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'vectors_count': self.vectors_count,
            'indexed_vectors_count': self.indexed_vectors_count,
            'points_count': self.points_count,
            'segments_count': self.segments_count,
            'config': self.config,
            'payload_schema': self.payload_schema
        } 