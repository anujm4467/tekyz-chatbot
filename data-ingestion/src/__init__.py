"""
Tekyz Data Ingestion Layer

A comprehensive data ingestion pipeline for the Tekyz Knowledge-Based Chatbot.
Provides web scraping, content processing, embedding generation, and vector database integration.
"""

# Scraper components
from .scraper import (
    TekyzScraper, URLDiscovery, ContentExtractor, ScrapingOrchestrator,
    ScrapingStats
)

# Processor components  
from .processor import (
    TextProcessor, TextChunk, ChunkingConfig, MetadataExtractor, 
    QualityController, ProcessingPipeline, ProcessingConfig
)

# Embedding components
from .embeddings import (
    EmbeddingGenerator, EmbeddingConfig
)

# Database components
from .database import (
    QdrantManager, VectorConfig, SearchResult
)

__all__ = [
    # Scraper exports
    'TekyzScraper', 'URLDiscovery', 'ContentExtractor', 'ScrapingOrchestrator',
    'ScrapingStats',
    
    # Processor exports
    'TextProcessor', 'TextChunk', 'ChunkingConfig', 'MetadataExtractor',
    'QualityController', 'ProcessingPipeline', 'ProcessingConfig',
    
    # Embedding exports
    'EmbeddingGenerator', 'EmbeddingConfig',
    
    # Database exports
    'QdrantManager', 'VectorConfig', 'SearchResult'
]

__version__ = "1.0.0"
__author__ = "Tekyz Development Team" 