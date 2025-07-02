"""
Vector Search Engine

This module handles semantic search operations with the Qdrant vector database.
"""

import json
import time
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict

import requests
from sentence_transformers import SentenceTransformer
from loguru import logger

# Add the data-ingestion path to sys.path to import existing models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data-ingestion'))

try:
    from src.database.qdrant_client import QdrantManager
    from src.database.models import VectorConfig
except ImportError:
    # Fallback if data-ingestion modules are not available
    QdrantManager = None
    VectorConfig = None

from ..models.data_models import SearchResult, ConfidenceLevel
from ..utils.config_manager import ConfigManager


class VectorSearchEngine:
    """
    Handles vector similarity search using Qdrant database.
    
    Features:
    - Semantic search with embedding generation
    - Result ranking and filtering
    - Metadata extraction and source attribution
    - Performance monitoring
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the vector search engine."""
        self.config = config_manager
        self.embedding_model = None
        self.base_url = f"http://{self.config.qdrant_host}:{self.config.qdrant_port}"
        
        logger.info(f"VectorSearchEngine initialized for collection: {self.config.qdrant_collection}")
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model lazily."""
        if self.embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {self.config.embedding_model}")
                self.embedding_model = SentenceTransformer(
                    self.config.embedding_model
                )
                logger.success("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        
        return self.embedding_model
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the input text."""
        try:
            model = self._load_embedding_model()
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector database."""
        try:
            # Check collections endpoint
            response = requests.get(f"{self.base_url}/collections", timeout=5)
            response.raise_for_status()
            
            collections_data = response.json()
            collections = [c['name'] for c in collections_data.get('result', {}).get('collections', [])]
            
            # Check if our collection exists
            collection_exists = self.config.qdrant_collection in collections
            
            if collection_exists:
                # Get collection info
                collection_response = requests.get(
                    f"{self.base_url}/collections/{self.config.qdrant_collection}",
                    timeout=5
                )
                collection_response.raise_for_status()
                collection_info = collection_response.json()['result']
                
                # Try to load embedding model to verify it works
                embedding_model_ready = False
                try:
                    self._load_embedding_model()
                    embedding_model_ready = True
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                
                return {
                    'status': 'healthy',
                    'qdrant_connection': True,
                    'database_accessible': True,
                    'embedding_model': embedding_model_ready,
                    'ready_for_search': embedding_model_ready and collection_info.get('points_count', 0) > 0,
                    'collections': collections,
                    'target_collection': self.config.qdrant_collection,
                    'collection_exists': True,
                    'points_count': collection_info.get('points_count', 0),
                    'vector_size': collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
                }
            else:
                return {
                    'status': 'degraded',
                    'qdrant_connection': True,
                    'database_accessible': False,
                    'embedding_model': False,
                    'ready_for_search': False,
                    'collections': collections,
                    'target_collection': self.config.qdrant_collection,
                    'collection_exists': False,
                    'error': f"Collection '{self.config.qdrant_collection}' not found"
                }
                
        except Exception as e:
            logger.error(f"Vector database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'qdrant_connection': False,
                'database_accessible': False,
                'embedding_model': False,
                'ready_for_search': False,
                'error': str(e)
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get detailed database statistics."""
        try:
            health = self.health_check()
            if health['status'] != 'healthy':
                return health
            
            return {
                'collection_name': self.config.qdrant_collection,
                'points_count': health['points_count'],
                'vector_dimension': health['vector_size'],
                'embedding_model': self.config.embedding_model,
                'status': 'ready'
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def search(
        self,
        query: str,
        limit: int = None,
        score_threshold: float = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search for the given query.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            metadata_filter: Optional metadata filtering conditions
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        start_time = time.time()
        
        try:
            # Use config defaults if not provided
            limit = limit or self.config.max_search_results
            score_threshold = score_threshold or self.config.similarity_threshold
            
            logger.info(f"Searching for: '{query}' (limit={limit}, threshold={score_threshold})")
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Prepare search request
            search_payload = {
                "vector": query_embedding,
                "limit": limit,
                "with_payload": True,
                "score_threshold": score_threshold
            }
            
            # Add filter if provided
            if metadata_filter:
                search_payload["filter"] = metadata_filter
            
            # Perform search
            response = requests.post(
                f"{self.base_url}/collections/{self.config.qdrant_collection}/points/search",
                headers={"Content-Type": "application/json"},
                json=search_payload,
                timeout=10
            )
            response.raise_for_status()
            
            search_data = response.json()
            
            if search_data.get('status') != 'ok':
                raise Exception(f"Search failed: {search_data}")
            
            # Convert results to SearchResult objects
            results = []
            for hit in search_data.get('result', []):
                try:
                    # Handle the data structure where text might be in metadata
                    payload = hit.get('payload', {})
                    metadata = payload.get('metadata', {})
                    
                    # Extract text from the primary location (based on actual data structure)
                    text_content = payload.get('text', '')  # This is the main field with contact info
                    
                    # Fallback to other possible locations if main field is empty
                    if not text_content:
                        text_content = (
                            payload.get('content') or
                            payload.get('clean_text') or
                            payload.get('processed_text') or
                            metadata.get('content') or
                            metadata.get('text') or 
                            metadata.get('clean_text') or
                            metadata.get('processed_text') or
                            payload.get('chunk_text') or
                            payload.get('document_text') or
                            str(payload.get('page_content', ''))
                        )
                    
                    # Handle case where content is a dict with 'text' field
                    if isinstance(text_content, dict):
                        text_content = (
                            text_content.get('text') or
                            text_content.get('content') or 
                            text_content.get('clean_text') or
                            str(text_content)
                        )
                    
                    # Ensure we have a string and handle any encoding issues
                    text_content = str(text_content) if text_content else "No content available"
                    
                    # Clean up common artifacts that might interfere with display
                    text_content = text_content.replace('\ufeff', '').replace('\u00a9', '©')
                    
                    # Create search result
                    search_result = SearchResult(
                        id=str(hit['id']),
                        text=text_content,
                        source_url=metadata.get('source_url', ''),
                        score=float(hit['score']),
                        metadata={
                            'word_count': metadata.get('word_count', 0),
                            'sentence_count': metadata.get('sentence_count', 0),
                            'text_length': metadata.get('text_length', 0),
                            'batch_index': metadata.get('batch_index', 0),
                            'chunk_index': metadata.get('chunk_index', 0),
                            'title': self._extract_title_from_url(metadata.get('source_url', '')),
                            'chunk_id': payload.get('chunk_id', str(hit['id']))
                        }
                    )
                    
                    results.append(search_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process search result: {e}")
                    continue
            
            # Apply ranking and boost factors
            results = self._rank_results(results, query)
            
            search_time = time.time() - start_time
            logger.info(f"Search completed: {len(results)} results in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL."""
        if not url:
            return "Unknown Source"
        
        try:
            # Extract the path part and clean it up
            path = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
            if path:
                # Convert URL-style text to readable title
                title = path.replace('-', ' ').replace('_', ' ')
                title = ' '.join(word.capitalize() for word in title.split())
                return title
            return "Tekyz Content"
        except:
            return "Tekyz Content"
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Apply additional ranking and boost factors to search results."""
        try:
            # Apply boost factors based on content type and source
            for result in results:
                boost_factor = 1.0
                
                # Boost based on content characteristics
                if result.metadata.get('word_count', 0) > 50:
                    boost_factor *= 1.1  # Prefer substantial content
                
                if result.metadata.get('sentence_count', 0) > 3:
                    boost_factor *= 1.05  # Prefer well-structured content
                
                # Boost based on URL patterns
                url_lower = result.source_url.lower()
                if any(keyword in url_lower for keyword in ['service', 'product', 'solution']):
                    boost_factor *= 1.15  # Boost service/product pages
                elif any(keyword in url_lower for keyword in ['about', 'team', 'company']):
                    boost_factor *= 1.1   # Boost company info pages
                elif any(keyword in url_lower for keyword in ['blog', 'article', 'interview']):
                    boost_factor *= 1.05  # Slight boost for content pages
                
                # Apply boost to score
                result.score *= boost_factor
            
            # Sort by boosted score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to rank results: {e}")
            return results 