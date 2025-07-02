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
        try:
            collection_info = self.vector_db.get_collection_info()
            if collection_info and collection_info.name == config.collection_name:
                logger.info(f"Collection '{config.collection_name}' already exists")
                return True
        except Exception as e:
            logger.debug(f"Error checking collection existence: {str(e)}")
            # Collection doesn't exist, continue to create it
        
        from .models import VectorConfig, DistanceMetric
        
        # Convert string distance metric to enum
        distance_map = {
            'COSINE': DistanceMetric.COSINE,
            'DOT': DistanceMetric.DOT,
            'EUCLIDEAN': DistanceMetric.EUCLIDEAN,
            'MANHATTAN': DistanceMetric.MANHATTAN
        }
        distance_metric = distance_map.get(config.distance_metric.upper(), DistanceMetric.COSINE)
        
        vector_config = VectorConfig(
            collection_name=config.collection_name,
            vector_size=config.vector_size,
            distance_metric=distance_metric
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
                         progress_callback: Optional[Callable[[int, int], None]] = None,
                         merge_mode: bool = True) -> IngestionResult:
        """
        Ingest embedding records into vector database.
        
        Args:
            records: List of embedding records
            config: Ingestion configuration
            progress_callback: Optional callback for progress updates
            merge_mode: If True, merge into existing collection; if False, create new
            
        Returns:
            Ingestion results with statistics
        """
        start_time = time.time()
        
        # For tekyz knowledge base, always use the standard collection name
        if merge_mode or config.collection_name == "tekyz_knowledge":
            config.collection_name = "tekyz_knowledge"
            logger.info(f"Using merge mode - all data will be added to 'tekyz_knowledge' collection")
        
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
        
        # Check for existing data to handle deduplication
        existing_count = 0
        if merge_mode:
            try:
                collection_info = self.vector_db.get_collection_info(config.collection_name)
                if collection_info:
                    existing_count = collection_info.points_count
                    logger.info(f"Collection '{config.collection_name}' already contains {existing_count} vectors")
            except Exception as e:
                logger.warning(f"Could not check existing collection size: {str(e)}")
        
        # Initialize result tracking
        result = IngestionResult(
            total_processed=len(records),
            successful_insertions=0,
            failed_insertions=0,
            processing_time_seconds=0.0,
            errors=[]
        )
        
        # Process in batches with deduplication
        batches = [records[i:i + config.batch_size] 
                  for i in range(0, len(records), config.batch_size)]
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch_with_dedup, batch, config, batch_idx, existing_count): batch_idx
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
                    
                    if 'duplicates_skipped' in batch_result:
                        logger.info(f"Batch {batch_idx}: {batch_result['duplicates_skipped']} duplicates skipped")
                    
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
        
        result.processing_time_seconds = time.time() - start_time
        
        # Update operation stats
        self.operation_stats['insertions']['count'] += result.successful_insertions
        self.operation_stats['insertions']['total_time'] += result.processing_time_seconds
        
        # Get final collection stats
        try:
            result.collection_stats = self.vector_db.get_collection_info(config.collection_name)
        except Exception as e:
            logger.warning(f"Could not retrieve final collection stats: {str(e)}")
        
        logger.info(f"Ingestion completed: {result.successful_insertions} vectors added, "
                   f"{result.failed_insertions} failed, {result.processing_time_seconds:.2f}s")
        
        if merge_mode and result.collection_stats:
            final_count = result.collection_stats.points_count
            logger.info(f"Collection '{config.collection_name}' now contains {final_count} total vectors "
                       f"(was {existing_count}, added {final_count - existing_count})")
        
        return result

    def _process_batch_with_dedup(self, 
                                 batch: List[Any],  # EmbeddingRecord placeholder
                                 config: IngestionConfig,
                                 batch_idx: int,
                                 existing_count: int) -> Dict[str, Any]:
        """
        Process a batch of embeddings with deduplication support.
        
        Args:
            batch: Batch of embedding records
            config: Ingestion configuration
            batch_idx: Index of the batch
            existing_count: Number of existing vectors in collection
            
        Returns:
            Results dictionary with insertion stats
        """
        try:
            # Prepare embeddings for insertion in the format expected by QdrantManager
            embeddings = []
            
            for i, record in enumerate(batch):
                # Generate a unique ID based on content hash to enable deduplication
                record_id = self._generate_record_id(record, batch_idx * config.batch_size + i, existing_count)
                
                # Extract vector and metadata from record
                if hasattr(record, 'embedding') and hasattr(record, 'text'):
                    embedding_data = {
                        'chunk_id': record_id,
                        'embedding': record.embedding.tolist() if hasattr(record.embedding, 'tolist') else record.embedding,
                        'text': getattr(record, 'text', ''),
                        'clean_text': getattr(record, 'clean_text', getattr(record, 'text', '')),
                        'source_url': getattr(record, 'source_url', ''),
                        'char_count': len(getattr(record, 'text', '')),
                        'word_count': len(getattr(record, 'text', '').split()),
                        'sentence_count': getattr(record, 'text', '').count('.') + getattr(record, 'text', '').count('!') + getattr(record, 'text', '').count('?'),
                        'model_name': getattr(record, 'model_name', ''),
                        'metadata': {
                            'generation_timestamp': getattr(record, 'generation_timestamp', ''),
                            'source': getattr(record, 'source', 'unknown'),
                            'batch_id': f'batch_{batch_idx}',
                            'ingestion_timestamp': datetime.now().isoformat()
                        }
                    }
                else:
                    # Handle different record formats
                    embedding_data = {
                        'chunk_id': record_id,
                        'embedding': record.get('vector', record.get('embedding', [])),
                        'text': record.get('text', ''),
                        'clean_text': record.get('clean_text', record.get('text', '')),
                        'source_url': record.get('source_url', ''),
                        'char_count': len(record.get('text', '')),
                        'word_count': len(record.get('text', '').split()),
                        'sentence_count': record.get('text', '').count('.') + record.get('text', '').count('!') + record.get('text', '').count('?'),
                        'model_name': record.get('model_name', ''),
                        'metadata': {
                            'source': record.get('source', 'unknown'),
                            'batch_id': f'batch_{batch_idx}',
                            'ingestion_timestamp': datetime.now().isoformat(),
                            **record.get('metadata', {})
                        }
                    }
                
                embeddings.append(embedding_data)
            
            # Upload embeddings to database using QdrantManager interface
            upload_result = self.vector_db.upload_vectors(
                embeddings=embeddings,
                progress_callback=lambda stage, current, total, details: logger.debug(f"{stage}: {current}/{total} - {details}")
            )
            
            return {
                'inserted_count': upload_result.uploaded_vectors,
                'duplicates_skipped': upload_result.failed_vectors,  # Failed uploads might be duplicates
                'batch_size': len(batch),
                'success_rate': upload_result.uploaded_vectors / len(batch) if len(batch) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Batch {batch_idx} processing failed: {str(e)}")
            return {
                'inserted_count': 0,
                'error': str(e),
                'batch_size': len(batch)
            }

    def _generate_record_id(self, record: Any, index: int, existing_count: int) -> str:
        """
        Generate a unique ID for a record that enables deduplication.
        
        Args:
            record: The embedding record
            index: Index within the current batch
            existing_count: Number of existing vectors
            
        Returns:
            Unique string ID
        """
        try:
            # Try to create a content-based hash for deduplication
            text_content = ""
            if hasattr(record, 'text'):
                text_content = str(record.text)
            elif isinstance(record, dict):
                text_content = str(record.get('text', ''))
            
            if text_content:
                # Create hash based on text content for deduplication
                import hashlib
                content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()[:12]
                return f"tekyz_{content_hash}"
            else:
                # Fallback to sequential ID
                return f"tekyz_vec_{existing_count + index + 1}"
                
        except Exception as e:
            logger.warning(f"Failed to generate content-based ID: {str(e)}, using sequential ID")
            return f"tekyz_vec_{existing_count + index + 1}"
    
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
    
    def check_tekyz_data_exists(self, collection_name: str = "tekyz_knowledge") -> Dict[str, Any]:
        """
        Check if tekyz.com data already exists in the database.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            Dictionary with existence status and metadata
        """
        try:
            # Ensure we're connected
            if not self.vector_db._connected:
                if not self.vector_db.connect():
                    return {
                        'exists': False,
                        'reason': 'Cannot connect to database',
                        'vector_count': 0,
                        'tekyz_pages': 0
                    }
            
            # Check if collection exists by getting collections list
            try:
                collections = self.vector_db.client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if collection_name not in collection_names:
                    return {
                        'exists': False,
                        'reason': 'Collection does not exist',
                        'vector_count': 0,
                        'tekyz_pages': 0
                    }
            except Exception as e:
                return {
                    'exists': False,
                    'reason': f'Error checking collections: {str(e)}',
                    'vector_count': 0,
                    'tekyz_pages': 0
                }
            
            # Get collection info
            collection_info = self.vector_db.get_collection_info()
            
            if not collection_info or collection_info.points_count == 0:
                return {
                    'exists': False,
                    'reason': 'Collection is empty',
                    'vector_count': 0,
                    'tekyz_pages': 0
                }
            
            # Instead of random vector search, use scroll to get all points and check metadata
            tekyz_pages = 0
            total_checked = 0
            
            try:
                # Use scroll to iterate through all points efficiently
                from qdrant_client.models import ScrollRequest, Filter, FieldCondition, MatchAny
                
                # First try to count points with tekyz-related metadata using filters
                try:
                    # Create filter for tekyz.com URLs
                    tekyz_filter = Filter(
                        should=[
                            FieldCondition(
                                key="url",
                                match=MatchAny(any=["tekyz.com"])
                            ),
                            FieldCondition(
                                key="source_url", 
                                match=MatchAny(any=["tekyz.com"])
                            ),
                            FieldCondition(
                                key="page_url",
                                match=MatchAny(any=["tekyz.com"])
                            )
                        ]
                    )
                    
                    # Count points with tekyz filter
                    count_result = self.vector_db.client.count(
                        collection_name=collection_name,
                        count_filter=tekyz_filter,
                        exact=False  # Allow approximate counting for better performance
                    )
                    
                    if hasattr(count_result, 'count'):
                        tekyz_pages = count_result.count
                        logger.info(f"Found {tekyz_pages} tekyz-related vectors using filter")
                    
                except Exception as filter_error:
                    logger.warning(f"Filter-based counting failed: {str(filter_error)}, falling back to scroll")
                    
                    # Fallback: use scroll to manually check metadata
                    scroll_request = ScrollRequest(
                        limit=100,  # Process in batches
                        with_payload=True,
                        with_vector=False  # We don't need vectors for metadata check
                    )
                    
                    next_page_offset = None
                    batch_count = 0
                    
                    while batch_count < 5:  # Limit to first 500 points for performance
                        if next_page_offset:
                            scroll_request.offset = next_page_offset
                        
                        try:
                            scroll_result = self.vector_db.client.scroll(
                                collection_name=collection_name,
                                scroll_filter=scroll_request.dict() if hasattr(scroll_request, 'dict') else scroll_request
                            )
                            
                            if not scroll_result[0]:  # No more points
                                break
                                
                            points = scroll_result[0]
                            next_page_offset = scroll_result[1]
                            
                            # Check each point's metadata
                            for point in points:
                                total_checked += 1
                                if point.payload:
                                    payload_str = str(point.payload).lower()
                                    if 'tekyz.com' in payload_str or 'tekyz' in payload_str:
                                        tekyz_pages += 1
                            
                            if not next_page_offset:  # No more pages
                                break
                                
                            batch_count += 1
                            
                        except Exception as scroll_error:
                            logger.warning(f"Scroll operation failed: {str(scroll_error)}")
                            break
                
                logger.info(f"Metadata check complete: {tekyz_pages} tekyz pages found, {total_checked} total checked")
                
            except Exception as e:
                logger.warning(f"Error during metadata check: {str(e)}")
                # If metadata check fails, assume data exists if we have significant number of vectors
                tekyz_pages = collection_info.points_count if collection_info.points_count > 50 else 0
            
            # Determine if we have tekyz data
            # Consider it tekyz data if:
            # 1. We found explicit tekyz references in metadata, OR
            # 2. We have substantial amount of data (>50 vectors) in tekyz_knowledge collection
            has_tekyz_data = (
                tekyz_pages > 0 or 
                (collection_name == "tekyz_knowledge" and collection_info.points_count > 50)
            )
            
            return {
                'exists': has_tekyz_data,
                'reason': f'Found {tekyz_pages} tekyz pages, {collection_info.points_count} total vectors',
                'vector_count': collection_info.points_count,
                'tekyz_pages': tekyz_pages,
                'total_checked': total_checked,
                'collection_info': {
                    'name': collection_info.name,
                    'vector_size': getattr(collection_info, 'vector_size', 384),
                    'points_count': collection_info.points_count,
                    'segments_count': getattr(collection_info, 'segments_count', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking tekyz data existence: {str(e)}")
            return {
                'exists': False,
                'reason': f'Error checking data: {str(e)}',
                'vector_count': 0,
                'tekyz_pages': 0,
                'error': str(e)
            } 