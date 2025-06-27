"""
Vector Database Manager

This module provides high-level management of vector database operations.
Handles batch processing, embedding integration, and collection lifecycle management.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from tqdm import tqdm

from .qdrant_client import QdrantManager
from .models import SearchResult, CollectionInfo
from embeddings.embedding_storage import EmbeddingStorage
from typing import Any as EmbeddingRecord  # Placeholder for now

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    collection_name: str
    vector_size: int
    distance_metric: str = "Cosine"
    batch_size: int = 100
    max_workers: int = 4
    enable_compression: bool = True
    backup_enabled: bool = True
    quality_threshold: float = 0.5


@dataclass
class IngestionResult:
    """Results from data ingestion process."""
    total_processed: int
    successful_insertions: int
    failed_insertions: int
    processing_time_seconds: float
    errors: List[str]
    collection_stats: Optional[CollectionInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        if self.collection_stats:
            result['collection_stats'] = self.collection_stats.to_dict()
        return result


@dataclass
class SearchConfig:
    """Configuration for vector search operations."""
    limit: int = 10
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    deduplicate_results: bool = True
    rerank_results: bool = False


class VectorDBManager:
    """
    High-level vector database management.
    
    Provides:
    - Batch ingestion with progress tracking
    - Collection lifecycle management
    - Search optimization and caching
    - Performance monitoring
    - Data validation and quality control
    """
    
    def __init__(self, 
                 vector_db: QdrantManager,
                 embedding_storage: Optional['EmbeddingStorage'] = None):
        """
        Initialize the vector database manager.
        
        Args:
            vector_db: Qdrant vector database manager
            embedding_storage: Optional embedding storage for backup/restore
        """
        self.vector_db = vector_db
        self.embedding_storage = embedding_storage
        
        # Performance tracking
        self.operation_stats = {
            'insertions': {'count': 0, 'total_time': 0.0},
            'searches': {'count': 0, 'total_time': 0.0},
            'errors': {'count': 0, 'last_error': None}
        }
        
        # Search cache (simple in-memory cache)
        self.search_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("Initialized VectorDBManager")
    
    def create_collection_if_not_exists(self, config: IngestionConfig) -> bool:
        """
        Create collection if it doesn't exist.
        
        Args:
            config: Ingestion configuration
            
        Returns:
            True if collection exists or was created successfully
        """
        if self.vector_db.collection_exists(config.collection_name):
            logger.info(f"Collection '{config.collection_name}' already exists")
            return True
        
        from .models import VectorConfig
        vector_config = VectorConfig(
            collection_name=config.collection_name,
            vector_size=config.vector_size,
            distance=config.distance_metric.upper()
        )
        success = self.vector_db.create_collection(vector_config)
        
        if success:
            logger.info(f"Created collection '{config.collection_name}'")
        else:
            logger.error(f"Failed to create collection '{config.collection_name}'")
        
        return success
    
    def ingest_embeddings(self, 
                         records: List[Any],  # EmbeddingRecord placeholder
                         config: IngestionConfig,
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> IngestionResult:
        """
        Ingest embedding records into vector database.
        
        Args:
            records: List of embedding records
            config: Ingestion configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Ingestion results with statistics
        """
        start_time = time.time()
        
        # Ensure collection exists
        if not self.create_collection_if_not_exists(config):
            return IngestionResult(
                total_processed=0,
                successful_insertions=0,
                failed_insertions=len(records),
                processing_time_seconds=0.0,
                errors=["Failed to create collection"]
            )
        
        logger.info(f"Starting ingestion of {len(records)} records into '{config.collection_name}'")
        
        # Initialize result tracking
        result = IngestionResult(
            total_processed=len(records),
            successful_insertions=0,
            failed_insertions=0,
            processing_time_seconds=0.0,
            errors=[]
        )
        
        # Process in batches
        batches = [records[i:i + config.batch_size] 
                  for i in range(0, len(records), config.batch_size)]
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, config, batch_idx): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            
            completed_batches = 0
            for future in tqdm(as_completed(future_to_batch), 
                             total=len(batches), 
                             desc="Ingesting batches"):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_result = future.result()
                    result.successful_insertions += batch_result['inserted_count']
                    
                    if 'error' in batch_result:
                        result.errors.append(f"Batch {batch_idx}: {batch_result['error']}")
                        result.failed_insertions += len(batches[batch_idx])
                    
                except Exception as e:
                    error_msg = f"Batch {batch_idx} failed: {str(e)}"
                    result.errors.append(error_msg)
                    result.failed_insertions += len(batches[batch_idx])
                    logger.error(error_msg)
                
                completed_batches += 1
                if progress_callback:
                    progress_callback(completed_batches, len(batches))
        
        # Calculate processing time
        result.processing_time_seconds = time.time() - start_time
        
        # Get final collection stats
        result.collection_stats = self.vector_db.get_collection_info(config.collection_name)
        
        # Update operation stats
        self.operation_stats['insertions']['count'] += result.successful_insertions
        self.operation_stats['insertions']['total_time'] += result.processing_time_seconds
        
        if result.errors:
            self.operation_stats['errors']['count'] += len(result.errors)
            self.operation_stats['errors']['last_error'] = result.errors[-1]
        
        logger.info(f"Ingestion completed: {result.successful_insertions}/{len(records)} successful")
        
        return result
    
    def _process_batch(self, 
                      batch: List[Any],  # EmbeddingRecord placeholder
                      config: IngestionConfig,
                      batch_idx: int) -> Dict[str, Any]:
        """
        Process a single batch of embedding records.
        
        Args:
            batch: Batch of embedding records
            config: Ingestion configuration
            batch_idx: Batch index for tracking
            
        Returns:
            Batch processing results
        """
        try:
            # Extract vectors, texts, and metadata
            vectors = [record.embedding for record in batch]
            texts = [record.text for record in batch]
            metadata = [record.metadata for record in batch]
            ids = [record.id for record in batch]
            
            # Prepare vectors for insertion
            vector_data = []
            for i, (vector, text, meta, vector_id) in enumerate(zip(vectors, texts, metadata, ids)):
                vector_data.append({
                    'id': vector_id,
                    'vector': vector.tolist() if hasattr(vector, 'tolist') else vector,
                    'payload': {
                        'text': text,
                        'metadata': meta or {}
                    }
                })
            
            # Insert into vector database
            upload_result = self.vector_db.upload_vectors(
                collection_name=config.collection_name,
                vectors=vector_data,
                batch_size=len(vector_data)
            )
            
            result = {
                'inserted_count': upload_result.uploaded_count,
                'failed_count': upload_result.failed_count
            }
            
            if upload_result.errors:
                result['error'] = '; '.join(upload_result.errors)
            
            logger.debug(f"Processed batch {batch_idx}: {result['inserted_count']} insertions")
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            return {'inserted_count': 0, 'error': str(e)}
    
    def search_similar(self, 
                      collection_name: str,
                      query_vector: np.ndarray,
                      config: SearchConfig) -> List[SearchResult]:
        """
        Search for similar vectors with advanced configuration.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search
            config: Search configuration
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(collection_name, query_vector, config)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug("Returning cached search result")
            return cached_result
        
        try:
            # Perform search
            results = self.vector_db.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
                limit=config.limit,
                score_threshold=config.score_threshold
            )
            
            # Post-process results
            if config.deduplicate_results:
                results = self._deduplicate_results(results)
            
            if config.rerank_results:
                results = self._rerank_results(results, query_vector)
            
            # Cache results
            self._cache_result(cache_key, results)
            
            # Update statistics
            search_time = time.time() - start_time
            self.operation_stats['searches']['count'] += 1
            self.operation_stats['searches']['total_time'] += search_time
            
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            error_msg = f"Search failed in collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            
            self.operation_stats['errors']['count'] += 1
            self.operation_stats['errors']['last_error'] = error_msg
            
            return []
    
    def batch_search(self, 
                    collection_name: str,
                    query_vectors: List[np.ndarray],
                    config: SearchConfig,
                    progress_callback: Optional[Callable[[int, int], None]] = None) -> List[List[SearchResult]]:
        """
        Perform batch search for multiple query vectors.
        
        Args:
            collection_name: Name of the collection
            query_vectors: List of query vectors
            config: Search configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of search results for each query
        """
        logger.info(f"Starting batch search for {len(query_vectors)} queries")
        
        results = []
        
        for i, query_vector in enumerate(tqdm(query_vectors, desc="Searching")):
            search_results = self.search_similar(
                collection_name=collection_name,
                query_vector=query_vector,
                config=config
            )
            results.append(search_results)
            
            if progress_callback:
                progress_callback(i + 1, len(query_vectors))
        
        logger.info(f"Batch search completed for {len(query_vectors)} queries")
        return results
    
    def _generate_cache_key(self, 
                           collection_name: str,
                           query_vector: np.ndarray,
                           config: SearchConfig) -> str:
        """Generate cache key for search results."""
        # Simple hash-based cache key
        vector_hash = hash(query_vector.tobytes())
        config_hash = hash(str(asdict(config)))
        return f"{collection_name}_{vector_hash}_{config_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search result if valid."""
        if cache_key in self.search_cache:
            cached_data = self.search_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['results']
            else:
                # Remove expired cache entry
                del self.search_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, results: List[SearchResult]):
        """Cache search results."""
        self.search_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        # Simple cache size management (keep last 1000 entries)
        if len(self.search_cache) > 1000:
            oldest_key = min(self.search_cache.keys(), 
                           key=lambda k: self.search_cache[k]['timestamp'])
            del self.search_cache[oldest_key]
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on text content."""
        seen_texts = set()
        deduplicated = []
        
        for result in results:
            if result.text not in seen_texts:
                seen_texts.add(result.text)
                deduplicated.append(result)
        
        if len(deduplicated) < len(results):
            logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
        
        return deduplicated
    
    def _rerank_results(self, 
                       results: List[SearchResult], 
                       query_vector: np.ndarray) -> List[SearchResult]:
        """Rerank search results using additional criteria."""
        # Simple implementation - could be enhanced with more sophisticated reranking
        # For now, just ensure results are sorted by score
        return sorted(results, key=lambda r: r.score, reverse=True)
    
    def get_collection_summary(self, collection_name: str) -> Dict[str, Any]:
        """
        Get comprehensive collection summary.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection summary with statistics
        """
        stats = self.vector_db.get_collection_info(collection_name)
        if not stats:
            return {'error': f"Collection '{collection_name}' not found"}
        
        return {
            'collection_name': collection_name,
            'stats': stats.to_dict(),
            'health_check': self.vector_db.get_health(),
            'last_updated': datetime.now().isoformat()
        }
    
    def optimize_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Optimize collection performance.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Optimization results
        """
        # For Qdrant, optimization is handled automatically
        # This method provides a placeholder for custom optimization logic
        
        logger.info(f"Optimizing collection '{collection_name}'")
        
        # Clear search cache for this collection
        cache_keys_to_remove = [
            key for key in self.search_cache.keys() 
            if key.startswith(f"{collection_name}_")
        ]
        
        for key in cache_keys_to_remove:
            del self.search_cache[key]
        
        return {
            'collection_name': collection_name,
            'optimization_type': 'cache_cleanup',
            'cache_entries_removed': len(cache_keys_to_remove),
            'timestamp': datetime.now().isoformat()
        }
    
    def backup_collection(self, 
                         collection_name: str,
                         backup_path: Union[str, Path],
                         include_vectors: bool = False) -> Dict[str, Any]:
        """
        Create collection backup.
        
        Args:
            collection_name: Name of the collection
            backup_path: Path to save backup
            include_vectors: Whether to include vector data
            
        Returns:
            Backup results
        """
        if include_vectors and self.embedding_storage:
            # Use embedding storage for comprehensive backup
            logger.info(f"Creating comprehensive backup for '{collection_name}'")
            # This would require loading all vectors from the collection
            # Implementation would depend on specific requirements
        
        # Use basic vector DB backup
        return self.vector_db.backup_collection(collection_name, backup_path)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the manager.
        
        Returns:
            Performance statistics
        """
        stats = dict(self.operation_stats)
        
        # Calculate averages
        if stats['insertions']['count'] > 0:
            stats['insertions']['avg_time'] = (
                stats['insertions']['total_time'] / stats['insertions']['count']
            )
        
        if stats['searches']['count'] > 0:
            stats['searches']['avg_time'] = (
                stats['searches']['total_time'] / stats['searches']['count']
            )
        
        # Add cache statistics
        stats['cache'] = {
            'size': len(self.search_cache),
            'hit_ratio': 0.0  # Would need to track hits/misses for accurate ratio
        }
        
        stats['timestamp'] = datetime.now().isoformat()
        
        return stats
    
    def clear_cache(self):
        """Clear search result cache."""
        cache_size = len(self.search_cache)
        self.search_cache.clear()
        logger.info(f"Cleared {cache_size} cache entries")
    
    def cleanup_old_collections(self, max_age_days: int = 30) -> Dict[str, Any]:
        """
        Clean up old collections based on age.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Cleanup results
        """
        # This would require collection metadata tracking
        # For now, return placeholder
        logger.info(f"Collection cleanup for age > {max_age_days} days")
        
        return {
            'cleaned_collections': 0,
            'max_age_days': max_age_days,
            'timestamp': datetime.now().isoformat()
        } 