"""
Qdrant Vector Database Client

This module provides a high-level interface for interacting with Qdrant vector database.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from .models import VectorConfig, SearchResult, UploadResult, CollectionInfo, DistanceMetric

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    High-level manager for Qdrant vector database operations.
    
    Features:
    - Connection management with automatic retries
    - Collection creation and management
    - Batch vector upload with progress tracking
    - Vector similarity search
    - Collection statistics and health monitoring
    """
    
    def __init__(self, config: VectorConfig = None):
        """
        Initialize the Qdrant manager.
        
        Args:
            config: Configuration for vector database operations
        """
        self.config = config or VectorConfig()
        self.client = None
        self._connected = False
        
        logger.info(f"QdrantManager initialized for collection: {self.config.collection_name}")
    
    def connect(self) -> bool:
        """
        Connect to Qdrant database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Qdrant at {self.config.host}:{self.config.port}")
            
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
                prefer_grpc=self.config.prefer_grpc,
                https=self.config.https,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            # Test connection
            collections = self.client.get_collections()
            self._connected = True
            
            logger.info(f"Connected to Qdrant successfully")
            logger.info(f"Available collections: {[c.name for c in collections.collections]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Qdrant database."""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
        
        self.client = None
        self._connected = False
        logger.info("Disconnected from Qdrant")
    
    @contextmanager
    def connection(self):
        """Context manager for automatic connection handling."""
        try:
            if not self._connected:
                self.connect()
            yield self
        finally:
            pass  # Keep connection alive for reuse
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create collection if it doesn't exist.
        
        Args:
            recreate: Whether to recreate the collection if it exists
            
        Returns:
            True if collection created/exists, False otherwise
        """
        if not self._connected:
            logger.error("Not connected to Qdrant")
            return False
        
        try:
            collection_name = self.config.collection_name
            
            # Check if collection exists
            collections = self.client.get_collections()
            existing_collections = [c.name for c in collections.collections]
            
            if collection_name in existing_collections:
                if recreate:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Collection {collection_name} already exists")
                    return True
            
            # Create collection
            logger.info(f"Creating collection: {collection_name}")
            
            # Convert distance metric
            distance_map = {
                DistanceMetric.COSINE: models.Distance.COSINE,
                DistanceMetric.DOT: models.Distance.DOT,
                DistanceMetric.EUCLIDEAN: models.Distance.EUCLID,
                DistanceMetric.MANHATTAN: models.Distance.MANHATTAN
            }
            
            distance = distance_map.get(self.config.distance_metric, models.Distance.COSINE)
            
            # Create vector configuration
            vectors_config = models.VectorParams(
                size=self.config.vector_size,
                distance=distance
            )
            
            # Create HNSW configuration
            hnsw_config = models.HnswConfigDiff(**self.config.hnsw_config)
            
            # Create optimizers configuration
            optimizers_config = models.OptimizersConfigDiff(**self.config.optimizers_config)
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config
            )
            
            logger.info(f"Collection {collection_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {self.config.collection_name}: {e}")
            return False
    
    def upload_vectors(
        self, 
        embeddings: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> UploadResult:
        """
        Upload vectors to the collection.
        
        Args:
            embeddings: List of embedding dictionaries with 'embedding' and metadata
            progress_callback: Optional callback for progress updates
            
        Returns:
            UploadResult with upload statistics
        """
        if not self._connected:
            logger.error("Not connected to Qdrant")
            return UploadResult(
                success=False,
                total_vectors=len(embeddings),
                uploaded_vectors=0,
                failed_vectors=len(embeddings),
                upload_time=0.0,
                errors=["Not connected to Qdrant"]
            )
        
        start_time = time.time()
        total_vectors = len(embeddings)
        uploaded_vectors = 0
        failed_vectors = 0
        errors = []
        
        logger.info(f"Uploading {total_vectors} vectors to collection {self.config.collection_name}")
        
        try:
            # Process in batches
            batch_size = self.config.batch_size
            
            for i in range(0, total_vectors, batch_size):
                batch_end = min(i + batch_size, total_vectors)
                batch = embeddings[i:batch_end]
                
                # Prepare points for upload
                points = []
                for j, emb_data in enumerate(batch):
                    try:
                        # Generate unique UUID for point ID (Qdrant requirement)
                        point_id = str(uuid.uuid4())
                        
                        # Prepare payload (metadata)
                        payload = {
                            'chunk_id': emb_data.get('chunk_id', point_id),
                            'text': emb_data.get('text', ''),
                            'clean_text': emb_data.get('clean_text', ''),
                            'source_url': emb_data.get('source_url', ''),
                            'char_count': emb_data.get('char_count', 0),
                            'word_count': emb_data.get('word_count', 0),
                            'sentence_count': emb_data.get('sentence_count', 0),
                            'metadata': emb_data.get('metadata', {}),
                            'model_name': emb_data.get('model_name', ''),
                            'embedding_dim': emb_data.get('embedding_dim', 0)
                        }
                        
                        # Create point
                        point = models.PointStruct(
                            id=point_id,
                            vector=emb_data['embedding'],
                            payload=payload
                        )
                        points.append(point)
                        
                    except Exception as e:
                        logger.warning(f"Failed to prepare point {i + j}: {e}")
                        failed_vectors += 1
                        errors.append(f"Point {i + j}: {str(e)}")
                
                # Upload batch
                if points:
                    try:
                        operation_info = self.client.upsert(
                            collection_name=self.config.collection_name,
                            points=points
                        )
                        
                        uploaded_vectors += len(points)
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                stage="vector_upload",
                                current=batch_end,
                                total=total_vectors,
                                details=f"Uploaded batch {i//batch_size + 1}"
                            )
                        
                        logger.debug(f"Uploaded batch {i//batch_size + 1}: {len(points)} vectors")
                        
                    except Exception as e:
                        logger.error(f"Failed to upload batch {i//batch_size + 1}: {e}")
                        failed_vectors += len(points)
                        errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error during vector upload: {e}")
            errors.append(f"Upload error: {str(e)}")
        
        upload_time = time.time() - start_time
        success = uploaded_vectors > 0 and failed_vectors == 0
        
        result = UploadResult(
            success=success,
            total_vectors=total_vectors,
            uploaded_vectors=uploaded_vectors,
            failed_vectors=failed_vectors,
            upload_time=upload_time,
            errors=errors
        )
        
        logger.info(f"Vector upload completed: {uploaded_vectors}/{total_vectors} successful "
                   f"in {upload_time:.2f}s")
        
        return result
    
    def search_vectors(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            
        Returns:
            List of SearchResult objects
        """
        if not self._connected:
            logger.error("Not connected to Qdrant")
            return []
        
        try:
            # Prepare filter
            query_filter = None
            if filter_conditions:
                # Convert filter conditions to Qdrant filter format
                # This is a simplified implementation
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to SearchResult objects
            results = []
            for hit in search_result:
                result = SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    payload=hit.payload or {},
                    vector=hit.vector
                )
                results.append(result)
            
            logger.debug(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """
        Get information about the collection.
        
        Returns:
            CollectionInfo object or None if error
        """
        if not self._connected:
            logger.error("Not connected to Qdrant")
            return None
        
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            
            return CollectionInfo(
                name=self.config.collection_name,
                vectors_count=collection_info.vectors_count or 0,
                indexed_vectors_count=collection_info.indexed_vectors_count or 0,
                points_count=collection_info.points_count or 0,
                segments_count=collection_info.segments_count or 0,
                config=collection_info.config.dict() if collection_info.config else {},
                payload_schema=collection_info.payload_schema or {}
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the database connection and collection.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            'connected': self._connected,
            'collection_exists': False,
            'collection_info': None,
            'errors': []
        }
        
        try:
            if not self._connected:
                health_status['errors'].append("Not connected to Qdrant")
                return health_status
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.config.collection_name in collection_names:
                health_status['collection_exists'] = True
                
                # Get collection info
                collection_info = self.get_collection_info()
                if collection_info:
                    health_status['collection_info'] = collection_info.to_dict()
            else:
                health_status['errors'].append(f"Collection {self.config.collection_name} does not exist")
            
        except Exception as e:
            health_status['errors'].append(f"Health check error: {str(e)}")
        
        return health_status
    
    def get_existing_content_hashes(self) -> set:
        """
        Get all existing content hashes from the collection in one batch operation.
        
        Returns:
            Set of existing content hashes
        """
        if not self._connected:
            logger.warning("Not connected to Qdrant, returning empty hash set")
            return set()
        
        try:
            existing_hashes = set()
            offset = None
            
            # Scroll through all points to get content hashes
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.config.collection_name,
                    offset=offset,
                    limit=100,  # Batch size
                    with_payload=["content_hash"],  # Only fetch content_hash field
                    with_vectors=False  # Don't fetch vectors for efficiency
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                # Extract content hashes
                for point in points:
                    if point.payload and 'content_hash' in point.payload:
                        existing_hashes.add(point.payload['content_hash'])
                
                # Continue scrolling if there are more points
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"Retrieved {len(existing_hashes)} existing content hashes from collection")
            return existing_hashes
            
        except Exception as e:
            logger.warning(f"Error getting existing content hashes: {e}")
            return set()

    def check_content_exists(self, content_hash: str) -> bool:
        """
        Check if content with given hash already exists in the collection.
        
        Args:
            content_hash: MD5 hash of the content text
            
        Returns:
            True if content exists, False otherwise
        """
        if not self._connected:
            logger.warning("Not connected to Qdrant, assuming content doesn't exist")
            return False
        
        try:
            # Use a more efficient search instead of scroll
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=[0.0] * 384,  # Dummy query vector since we're filtering by hash
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_hash",
                            match=models.MatchValue(value=content_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            # If we found any results, content exists
            return len(search_results) > 0
            
        except Exception as e:
            logger.warning(f"Error checking content existence: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self._connected:
            return {"error": "Not connected to Qdrant"}
        
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            
            # Get count of points
            count_result = self.client.count(
                collection_name=self.config.collection_name,
                exact=True
            )
            
            return {
                "points_count": count_result.count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                },
                "status": collection_info.status.value,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": collection_info.indexed_vectors_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup on destruction."""
        if self._connected:
            self.disconnect() 