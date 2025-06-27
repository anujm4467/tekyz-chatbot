"""
Integration Pipeline

This module orchestrates the complete data ingestion pipeline from web scraping
through content processing, embedding generation, and vector database storage.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Generator
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from tqdm import tqdm

from scraper.orchestrator import ScrapingOrchestrator
from processor import ProcessingPipeline, ProcessingConfig
from embeddings import (
    ModelManager, TextPreprocessor, EmbeddingGenerator, 
    EmbeddingValidator, EmbeddingStorage, EmbeddingRecord
)
from .vector_db_manager import VectorDBManager, IngestionConfig, IngestionResult
from .qdrant_client import QdrantVectorDB

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    # Processing configuration  
    processing: ProcessingConfig
    
    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    
    # Vector DB configuration
    collection_name: str = "tekyz_knowledge"
    vector_db_host: str = "localhost"
    vector_db_port: int = 6333
    
    # Pipeline configuration
    max_concurrent_workers: int = 4
    save_intermediate_results: bool = True
    validate_embeddings: bool = True
    backup_data: bool = True


@dataclass
class PipelineResult:
    """Complete pipeline execution results."""
    total_urls_processed: int
    total_content_extracted: int
    total_chunks_created: int
    total_embeddings_generated: int
    total_vectors_stored: int
    
    processing_times: Dict[str, float]
    success_rates: Dict[str, float]
    errors: List[str]
    
    # Detailed results from each stage
    scraping_results: Optional[Dict[str, Any]] = None
    processing_results: Optional[Dict[str, Any]] = None
    embedding_results: Optional[Dict[str, Any]] = None
    ingestion_results: Optional[IngestionResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        if self.ingestion_results:
            result['ingestion_results'] = self.ingestion_results.to_dict()
        return result


class IntegrationPipeline:
    """
    Complete data ingestion pipeline orchestrator.
    
    Orchestrates:
    1. Web scraping and content extraction
    2. Content processing and chunking
    3. Embedding generation and validation
    4. Vector database storage and indexing
    """
    
    def __init__(self, 
                 config: PipelineConfig,
                 storage_dir: Optional[Path] = None):
        """
        Initialize the integration pipeline.
        
        Args:
            config: Pipeline configuration
            storage_dir: Optional directory for intermediate storage
        """
        self.config = config
        self.storage_dir = storage_dir or Path("data/pipeline_storage")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Pipeline state tracking
        self.pipeline_state = {
            'current_stage': None,
            'start_time': None,
            'stage_times': {},
            'intermediate_results': {}
        }
        
        logger.info("Initialized IntegrationPipeline")
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # Web scraper
        self.web_scraper = ScrapingOrchestrator()
        
        # Content processor
        self.content_processor = ProcessingPipeline(self.config.processing)
        
        # Embedding components
        self.model_manager = ModelManager()
        self.text_preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator()
        self.embedding_validator = EmbeddingValidator()
        
        # Storage components
        self.embedding_storage = EmbeddingStorage(self.storage_dir / "embeddings")
        
        # Vector database
        self.vector_db = QdrantVectorDB(
            host=self.config.vector_db_host,
            port=self.config.vector_db_port
        )
        
        self.vector_db_manager = VectorDBManager(
            vector_db=self.vector_db,
            embedding_storage=self.embedding_storage
        )
        
        logger.info("Initialized all pipeline components")
    
    def run_complete_pipeline(self, 
                             urls: List[str],
                             progress_callback: Optional[Callable[[str, int, int], None]] = None) -> PipelineResult:
        """
        Run the complete data ingestion pipeline.
        
        Args:
            urls: List of URLs to process
            progress_callback: Optional callback for progress updates (stage, current, total)
            
        Returns:
            Complete pipeline results
        """
        logger.info(f"Starting complete pipeline for {len(urls)} URLs")
        
        # Initialize result tracking
        result = PipelineResult(
            total_urls_processed=0,
            total_content_extracted=0,
            total_chunks_created=0,
            total_embeddings_generated=0,
            total_vectors_stored=0,
            processing_times={},
            success_rates={},
            errors=[]
        )
        
        self.pipeline_state['start_time'] = time.time()
        
        try:
            # Stage 1: Web Scraping
            scraped_content = self._run_scraping_stage(urls, progress_callback)
            result.scraping_results = scraped_content
            result.total_urls_processed = len(urls)
            result.total_content_extracted = len(scraped_content.get('extracted_content', []))
            
            if not scraped_content.get('extracted_content'):
                result.errors.append("No content extracted from scraping stage")
                return result
            
            # Stage 2: Content Processing
            processed_content = self._run_processing_stage(
                scraped_content['extracted_content'], 
                progress_callback
            )
            result.processing_results = processed_content
            result.total_chunks_created = len(processed_content.get('processed_chunks', []))
            
            if not processed_content.get('processed_chunks'):
                result.errors.append("No chunks created from processing stage")
                return result
            
            # Stage 3: Embedding Generation
            embeddings = self._run_embedding_stage(
                processed_content['processed_chunks'],
                progress_callback
            )
            result.embedding_results = embeddings
            result.total_embeddings_generated = len(embeddings.get('embedding_records', []))
            
            if not embeddings.get('embedding_records'):
                result.errors.append("No embeddings generated")
                return result
            
            # Stage 4: Vector Database Ingestion
            ingestion_result = self._run_ingestion_stage(
                embeddings['embedding_records'],
                progress_callback
            )
            result.ingestion_results = ingestion_result
            result.total_vectors_stored = ingestion_result.successful_insertions
            
            # Calculate success rates
            result.success_rates = self._calculate_success_rates(result)
            
            # Record processing times
            result.processing_times = dict(self.pipeline_state['stage_times'])
            
            logger.info("Complete pipeline execution finished successfully")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _run_scraping_stage(self, 
                           urls: List[str],
                           progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Run the web scraping stage."""
        self.pipeline_state['current_stage'] = 'scraping'
        stage_start = time.time()
        
        logger.info("Starting scraping stage")
        
        if progress_callback:
            progress_callback('scraping', 0, len(urls))
        
        try:
            # Use web scraper to extract content
            results = self.web_scraper.scrape_urls(urls)
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_intermediate_results('scraping', results)
            
            stage_time = time.time() - stage_start
            self.pipeline_state['stage_times']['scraping'] = stage_time
            
            logger.info(f"Scraping stage completed in {stage_time:.2f}s")
            
            if progress_callback:
                progress_callback('scraping', len(urls), len(urls))
            
            return results
            
        except Exception as e:
            logger.error(f"Scraping stage failed: {str(e)}")
            raise
    
    def _run_processing_stage(self, 
                             content_items: List[Dict[str, Any]],
                             progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Run the content processing stage."""
        self.pipeline_state['current_stage'] = 'processing'
        stage_start = time.time()
        
        logger.info("Starting processing stage")
        
        if progress_callback:
            progress_callback('processing', 0, len(content_items))
        
        try:
            # Process content items into chunks
            all_chunks = []
            
            for i, content_item in enumerate(content_items):
                # Extract text content
                text = content_item.get('content', '')
                if not text:
                    continue
                
                # Process through content pipeline
                chunks = self.content_processor.process_content(
                    text=text,
                    metadata={
                        'url': content_item.get('url', ''),
                        'title': content_item.get('title', ''),
                        'source': 'web_scraping',
                        **content_item.get('metadata', {})
                    }
                )
                
                all_chunks.extend(chunks)
                
                if progress_callback:
                    progress_callback('processing', i + 1, len(content_items))
            
            results = {
                'processed_chunks': all_chunks,
                'total_chunks': len(all_chunks),
                'source_items': len(content_items)
            }
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_intermediate_results('processing', results)
            
            stage_time = time.time() - stage_start
            self.pipeline_state['stage_times']['processing'] = stage_time
            
            logger.info(f"Processing stage completed in {stage_time:.2f}s")
            logger.info(f"Created {len(all_chunks)} chunks from {len(content_items)} content items")
            
            return results
            
        except Exception as e:
            logger.error(f"Processing stage failed: {str(e)}")
            raise
    
    def _run_embedding_stage(self, 
                            chunks: List[Dict[str, Any]],
                            progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Run the embedding generation stage."""
        self.pipeline_state['current_stage'] = 'embedding'
        stage_start = time.time()
        
        logger.info("Starting embedding stage")
        
        if progress_callback:
            progress_callback('embedding', 0, len(chunks))
        
        try:
            # Load embedding model
            model = self.model_manager.load_model(self.config.embedding_model)
            
            # Prepare texts for embedding
            texts = [chunk.get('text', '') for chunk in chunks]
            
            # Preprocess texts
            preprocessed_texts = self.text_preprocessor.preprocess_batch(texts)
            
            # Generate embeddings
            embeddings_data = self.embedding_generator.generate_embeddings(
                texts=preprocessed_texts,
                model=model,
                batch_size=self.config.embedding_batch_size
            )
            
            # Create embedding records
            embedding_records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_data['embeddings'])):
                record = EmbeddingRecord(
                    id=chunk.get('id', f'chunk_{i}'),
                    text=chunk.get('text', ''),
                    embedding=embedding,
                    metadata=chunk.get('metadata', {}),
                    timestamp=datetime.now().isoformat(),
                    source='pipeline',
                    embedding_model=self.config.embedding_model,
                    embedding_dimensions=len(embedding)
                )
                embedding_records.append(record)
                
                if progress_callback:
                    progress_callback('embedding', i + 1, len(chunks))
            
            # Validate embeddings if configured
            validation_results = None
            if self.config.validate_embeddings:
                validation_results = self.embedding_validator.validate_embeddings(
                    embeddings=[record.embedding for record in embedding_records],
                    texts=[record.text for record in embedding_records]
                )
            
            # Store embeddings if configured
            storage_results = None
            if self.config.backup_data:
                storage_results = self.embedding_storage.store_embeddings(
                    records=embedding_records,
                    format="pickle",
                    compress=True,
                    batch_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            
            results = {
                'embedding_records': embedding_records,
                'embeddings_data': embeddings_data,
                'validation_results': validation_results,
                'storage_results': storage_results,
                'total_embeddings': len(embedding_records)
            }
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                # Don't save the actual embeddings in intermediate results (too large)
                intermediate_results = {
                    'total_embeddings': len(embedding_records),
                    'embedding_dimensions': embedding_records[0].embedding_dimensions if embedding_records else 0,
                    'validation_results': validation_results,
                    'storage_results': storage_results
                }
                self._save_intermediate_results('embedding', intermediate_results)
            
            stage_time = time.time() - stage_start
            self.pipeline_state['stage_times']['embedding'] = stage_time
            
            logger.info(f"Embedding stage completed in {stage_time:.2f}s")
            logger.info(f"Generated {len(embedding_records)} embeddings")
            
            return results
            
        except Exception as e:
            logger.error(f"Embedding stage failed: {str(e)}")
            raise
    
    def _run_ingestion_stage(self, 
                            embedding_records: List[EmbeddingRecord],
                            progress_callback: Optional[Callable]) -> IngestionResult:
        """Run the vector database ingestion stage."""
        self.pipeline_state['current_stage'] = 'ingestion'
        stage_start = time.time()
        
        logger.info("Starting ingestion stage")
        
        if progress_callback:
            progress_callback('ingestion', 0, len(embedding_records))
        
        try:
            # Configure ingestion
            ingestion_config = IngestionConfig(
                collection_name=self.config.collection_name,
                vector_size=embedding_records[0].embedding_dimensions if embedding_records else 384,
                distance_metric="Cosine",
                batch_size=100,
                max_workers=self.config.max_concurrent_workers
            )
            
            # Create progress callback for ingestion
            def ingestion_progress(completed, total):
                if progress_callback:
                    progress_callback('ingestion', completed * len(embedding_records) // total, len(embedding_records))
            
            # Ingest embeddings
            result = self.vector_db_manager.ingest_embeddings(
                records=embedding_records,
                config=ingestion_config,
                progress_callback=ingestion_progress
            )
            
            stage_time = time.time() - stage_start
            self.pipeline_state['stage_times']['ingestion'] = stage_time
            
            logger.info(f"Ingestion stage completed in {stage_time:.2f}s")
            logger.info(f"Stored {result.successful_insertions}/{len(embedding_records)} vectors")
            
            if progress_callback:
                progress_callback('ingestion', len(embedding_records), len(embedding_records))
            
            return result
            
        except Exception as e:
            logger.error(f"Ingestion stage failed: {str(e)}")
            raise
    
    def _save_intermediate_results(self, stage: str, results: Dict[str, Any]):
        """Save intermediate results to disk."""
        try:
            results_file = self.storage_dir / f"{stage}_results.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.debug(f"Saved {stage} intermediate results to {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results for {stage}: {str(e)}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _calculate_success_rates(self, result: PipelineResult) -> Dict[str, float]:
        """Calculate success rates for each stage."""
        success_rates = {}
        
        # Scraping success rate
        if result.total_urls_processed > 0:
            success_rates['scraping'] = result.total_content_extracted / result.total_urls_processed
        
        # Processing success rate  
        if result.total_content_extracted > 0:
            success_rates['processing'] = result.total_chunks_created / result.total_content_extracted
        
        # Embedding success rate
        if result.total_chunks_created > 0:
            success_rates['embedding'] = result.total_embeddings_generated / result.total_chunks_created
        
        # Ingestion success rate
        if result.total_embeddings_generated > 0:
            success_rates['ingestion'] = result.total_vectors_stored / result.total_embeddings_generated
        
        # Overall success rate
        if result.total_urls_processed > 0:
            success_rates['overall'] = result.total_vectors_stored / result.total_urls_processed
        
        return success_rates
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'current_stage': self.pipeline_state['current_stage'],
            'start_time': self.pipeline_state.get('start_time'),
            'stage_times': dict(self.pipeline_state['stage_times']),
            'elapsed_time': time.time() - self.pipeline_state['start_time'] if self.pipeline_state['start_time'] else 0,
            'components_initialized': True,
            'storage_dir': str(self.storage_dir)
        }
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate result files."""
        try:
            for file_path in self.storage_dir.glob("*_results.json"):
                file_path.unlink()
            logger.info("Cleaned up intermediate result files")
        except Exception as e:
            logger.warning(f"Failed to cleanup intermediate files: {str(e)}")
    
    def load_intermediate_results(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load intermediate results from a specific stage."""
        try:
            results_file = self.storage_dir / f"{stage}_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load intermediate results for {stage}: {str(e)}")
        return None 