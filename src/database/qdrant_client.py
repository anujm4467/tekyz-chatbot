import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from loguru import logger

from .models import VectorConfig, SearchResult, UploadResult, CollectionInfo

class QdrantManager:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize Qdrant client manager."""
        self.host = host
        self.port = port
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant."""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible."""
        try:
            self.client.get_collections()
            logger.info("Qdrant health check passed")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def create_collection(self, config: VectorConfig) -> bool:
        """Create a new collection."""
        try:
            # Convert string distance to qdrant Distance enum
            distance_map = {
                "COSINE": Distance.COSINE,
                "DOT": Distance.DOT,
                "EUCLIDEAN": Distance.EUCLIDEAN,
                "MANHATTAN": Distance.MANHATTAN
            }
            
            distance = distance_map.get(config.distance.upper(), Distance.COSINE)
            
            self.client.create_collection(
                collection_name=config.collection_name,
                vectors_config=VectorParams(
                    size=config.vector_size,
                    distance=distance
                )
            )
            logger.info(f"Created collection '{config.collection_name}' with {config.vector_size}D vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get collection information."""
        try:
            info = self.client.get_collection(collection_name)
            return CollectionInfo(
                name=collection_name,
                vector_size=info.config.params.vectors.size,
                points_count=info.points_count,
                distance=info.config.params.vectors.distance.name
            )
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

    def upload_vectors(self, collection_name: str, vectors: List[Dict], batch_size: int = 100, 
                      progress_callback: Optional[callable] = None) -> UploadResult:
        """Upload vectors to collection with proper UUID-based IDs."""
        try:
            total_vectors = len(vectors)
            uploaded_count = 0
            failed_count = 0
            errors = []
            
            logger.info(f"Starting upload of {total_vectors} vectors to '{collection_name}'")
            
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    # Convert vectors to PointStruct with UUID IDs
                    points = []
                    for vector_data in batch:
                        # Generate a UUID for Qdrant point ID
                        point_id = str(uuid.uuid4())
                        
                        # Store original chunk ID in payload for reference
                        payload = vector_data.get('payload', {})
                        payload['original_chunk_id'] = vector_data.get('id', 'unknown')
                        
                        point = PointStruct(
                            id=point_id,
                            vector=vector_data['vector'],
                            payload=payload
                        )
                        points.append(point)
                    
                    # Upload batch
                    operation_info = self.client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    
                    uploaded_count += len(batch)
                    logger.info(f"Successfully uploaded batch {batch_num} ({len(batch)} vectors)")
                    
                    if progress_callback:
                        progress_callback(uploaded_count, total_vectors)
                        
                except Exception as e:
                    failed_count += len(batch)
                    error_msg = f"Batch {batch_num}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"Failed to upload batch {batch_num}: {e}")
                    continue
            
            success_rate = (uploaded_count / total_vectors) * 100 if total_vectors > 0 else 0
            
            return UploadResult(
                total_vectors=total_vectors,
                uploaded_count=uploaded_count,
                failed_count=failed_count,
                success_rate=success_rate,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Upload operation failed: {e}")
            return UploadResult(
                total_vectors=len(vectors),
                uploaded_count=0,
                failed_count=len(vectors),
                success_rate=0.0,
                errors=[str(e)]
            )

    def search_vectors(self, collection_name: str, query_vector: List[float], 
                      limit: int = 10, score_threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar vectors in collection."""
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for hit in search_result:
                result = SearchResult(
                    id=hit.id,
                    score=hit.score,
                    payload=hit.payload or {}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [] 