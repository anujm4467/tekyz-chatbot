"""
Pipeline Manager for Web Application

Manages the data ingestion pipeline execution with real-time updates
and error handling for web applications.
"""

import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime
import json
import logging

from ui.progress_tracker import ProgressTracker
from ui.error_handler import ErrorHandler
from ui.file_handlers import WordDocumentProcessor
from processor.pipeline import ProcessingPipeline, ProcessingConfig
from processor.text_processor import ChunkingConfig
from embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from database.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages pipeline execution for web applications
    
    Features:
    - Real-time progress tracking
    - Error handling and retry mechanisms
    - Integration with existing pipeline components
    - Thread-safe execution
    """
    
    def __init__(self):
        """Initialize the pipeline manager"""
        self.progress_tracker = ProgressTracker()
        self.error_handler = ErrorHandler()
        
        # Pipeline components
        self.processing_pipeline: Optional[ProcessingPipeline] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.qdrant_manager: Optional[QdrantManager] = None
        
        # Execution state
        self.is_running = False
        self.current_job_id: Optional[str] = None
        self.executor_thread: Optional[threading.Thread] = None
        
        # Results storage
        self.last_results: Optional[Dict[str, Any]] = None
        
        # Configuration
        self.default_config = {
            'chunk_size': 1024,
            'overlap_size': 50,
            'quality_threshold': 0.6,
            'max_workers': 4,
            'batch_size': 50,
            'embedding_model': 'all-MiniLM-L6-v2',
            'device': 'auto'
        }
    
    def start_pipeline(self, uploaded_files: List, urls: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Start the data ingestion pipeline
        
        Args:
            uploaded_files: List of uploaded file objects
            urls: List of URLs to process
            config: Pipeline configuration
        """
        logger.info(f"PipelineManager.start_pipeline() called")
        logger.info(f"uploaded_files count: {len(uploaded_files) if uploaded_files else 0}")
        logger.info(f"urls count: {len(urls) if urls else 0}")
        
        if self.is_running:
            error_msg = "Pipeline already running"
            logger.error(error_msg)
            self.error_handler.add_error(error_msg, "pipeline_manager")
            return False
        
        # Use default config if none provided
        if config is None:
            config = self.default_config.copy()
        
        # Generate job ID
        self.current_job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Generated job_id: {self.current_job_id}")
        
        # Prepare input data
        input_data = self._prepare_input_data(uploaded_files, urls)
        logger.info(f"Prepared input data count: {len(input_data) if input_data else 0}")
        
        if not input_data:
            error_msg = "No valid input data provided"
            logger.error(error_msg)
            self.error_handler.add_error(error_msg, "pipeline_manager")
            return False
        
        # Start pipeline in background thread
        self.is_running = True
        self.progress_tracker.start_pipeline()
        
        self.executor_thread = threading.Thread(
            target=self._execute_pipeline,
            args=(input_data, config),
            daemon=True
        )
        
        self.executor_thread.start()
        logger.info(f"Pipeline started successfully, thread alive: {self.executor_thread.is_alive()}")
        
        return True
    
    def stop_pipeline(self):
        """Stop the currently running pipeline"""
        if self.is_running:
            self.is_running = False
            self.progress_tracker.stop_pipeline(success=False)
            logger.info("Pipeline stopped by user")
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'is_running': self.is_running,
            'job_id': self.current_job_id,
            'progress': self.progress_tracker.get_overall_progress(),
            'current_step': self.progress_tracker.metrics.current_step,
            'estimated_completion': self.progress_tracker.get_estimated_completion_time(),
            'errors': self.error_handler.get_recent_errors()
        }
    
    def _prepare_input_data(self, uploaded_files: List, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Prepare input data from uploaded files and URLs
        
        Args:
            uploaded_files: List of uploaded file objects
            urls: List of URLs to process
            
        Returns:
            List of prepared input data
        """
        input_data = []
        
        # Process uploaded files
        if uploaded_files:
            self.progress_tracker.add_log(f"Processing {len(uploaded_files)} uploaded files", "info")
            
            try:
                file_results = WordDocumentProcessor.process_multiple_files(uploaded_files)
                
                for result in file_results:
                    if result['success']:
                        input_data.append({
                            'type': 'document',
                            'source': result['metadata']['filename'],
                            'content': result['text'],
                            'metadata': result['metadata']
                        })
                    else:
                        self.error_handler.add_error(
                            f"Failed to process file {result['metadata']['filename']}: {result['error']}",
                            "file_processing"
                        )
            except Exception as e:
                self.error_handler.add_error(f"Error processing files: {str(e)}", "file_processing")
        
        # Process URLs
        if urls:
            self.progress_tracker.add_log(f"Preparing {len(urls)} URLs for processing", "info")
            
            # For URLs, we'll create placeholder entries
            # The actual web scraping would be implemented here
            for url in urls:
                if url.strip():
                    input_data.append({
                        'type': 'url',
                        'source': url.strip(),
                        'content': None,  # Will be populated by scraper
                        'metadata': {'url': url.strip()}
                    })
        
        return input_data
    
    def _execute_pipeline(self, input_data: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Execute the complete pipeline in a background thread
        
        Args:
            input_data: Prepared input data
            config: Pipeline configuration
        """
        try:
            logger.info("Starting pipeline execution")
            self.progress_tracker.add_log("Pipeline execution started", "info")
            
            # Step 1: Document Processing
            self.progress_tracker.update_step("document_processing", 0, len(input_data))
            processed_data = self._execute_document_processing(input_data, config)
            
            if not processed_data:
                raise Exception("No documents were processed successfully")
            
            # Step 2: Text Chunking
            self.progress_tracker.update_step("text_chunking", 0, len(processed_data))
            chunked_data = self._execute_text_chunking(processed_data, config)
            
            if not chunked_data:
                raise Exception("No text chunks were created")
            
            # Step 3: Embedding Generation
            self.progress_tracker.update_step("embedding_generation", 0, len(chunked_data))
            embeddings_data = self._execute_embedding_generation(chunked_data, config)
            
            if not embeddings_data:
                raise Exception("No embeddings were generated")
            
            # Step 4: Database Upload
            self.progress_tracker.update_step("database_upload", 0, len(embeddings_data))
            upload_results = self._execute_database_upload(embeddings_data, config)
            
            # Step 5: Validation
            self.progress_tracker.update_step("validation", 0, 1)
            validation_results = self._execute_validation(upload_results, config)
            
            # Complete successfully
            self.last_results = {
                'success': True,
                'processed_documents': len(processed_data),
                'total_chunks': len(chunked_data),
                'embeddings_created': len(embeddings_data),
                'upload_results': upload_results,
                'validation_results': validation_results,
                'completion_time': datetime.now().isoformat()
            }
            
            self.progress_tracker.stop_pipeline(success=True)
            self.progress_tracker.add_log("Pipeline completed successfully!", "success")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            self.error_handler.add_error(str(e), "pipeline_execution")
            self.progress_tracker.stop_pipeline(success=False)
            
            self.last_results = {
                'success': False,
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
        
        finally:
            self.is_running = False
            logger.info("Pipeline execution completed")
    
    def _execute_document_processing(self, input_data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute document processing step"""
        processed_data = []
        
        for i, item in enumerate(input_data):
            if not self.is_running:
                break
                
            try:
                # For documents, content is already extracted
                if item['type'] == 'document':
                    processed_data.append(item)
                
                # For URLs, we would implement web scraping here
                elif item['type'] == 'url':
                    # Placeholder for URL processing
                    self.progress_tracker.add_log(f"URL processing not yet implemented: {item['source']}", "warning")
                
                self.progress_tracker.update_step("document_processing", i + 1, len(input_data))
                
            except Exception as e:
                self.error_handler.add_error(f"Error processing {item['source']}: {str(e)}", "document_processing")
        
        return processed_data
    
    def _execute_text_chunking(self, processed_data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute text chunking step"""
        chunked_data = []
        
        # Create chunking configuration
        chunking_config = ChunkingConfig(
            max_chunk_size=config.get('chunk_size', 1024),
            overlap_size=config.get('overlap_size', 50),
            min_chunk_size=100,
            split_by_sentences=True,
            preserve_paragraphs=True,
            respect_word_boundaries=True
        )
        
        # Create processing configuration
        processing_config = ProcessingConfig(
            chunking=chunking_config,
            min_quality_score=config.get('quality_threshold', 0.6),
            max_workers=config.get('max_workers', 4),
            batch_size=config.get('batch_size', 50),
            output_dir=Path("data/processed")
        )
        
        # Initialize processing pipeline
        self.processing_pipeline = ProcessingPipeline(processing_config)
        
        for i, item in enumerate(processed_data):
            if not self.is_running:
                break
                
            try:
                # Create chunks for each document
                content = item['content']
                if content:
                    # Use the processing pipeline's chunking functionality
                    chunks = self.processing_pipeline.text_processor.chunk_text(content, chunking_config)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_data = {
                            'chunk_id': f"{item['source']}_{chunk_idx}",
                            'source': item['source'],
                            'content': chunk.text,
                            'metadata': {
                                **item['metadata'],
                                'chunk_index': chunk_idx,
                                'chunk_start': chunk.start_index,
                                'chunk_end': chunk.end_index
                            }
                        }
                        chunked_data.append(chunk_data)
                
                self.progress_tracker.update_step("text_chunking", i + 1, len(processed_data))
                
            except Exception as e:
                self.error_handler.add_error(f"Error chunking {item['source']}: {str(e)}", "text_chunking")
        
        return chunked_data
    
    def _execute_embedding_generation(self, chunked_data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute embedding generation step"""
        embeddings_data = []
        
        # Create embedding configuration
        embedding_config = EmbeddingConfig(
            model_name=config.get('embedding_model', 'all-MiniLM-L6-v2'),
            batch_size=config.get('batch_size', 32),
            device=config.get('device', 'auto'),
            cache_embeddings=True,
            cache_dir=Path("data/embeddings/cache")
        )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        
        try:
            # Prepare texts for embedding
            texts = [item['content'] for item in chunked_data]
            
            # Generate embeddings
            embedding_result = self.embedding_generator.generate_embeddings(
                texts,
                progress_callback=lambda current, total: self.progress_tracker.update_step(
                    "embedding_generation", current, total
                )
            )
            
            # Combine embeddings with metadata
            for i, embedding in enumerate(embedding_result.embeddings):
                if i < len(chunked_data):
                    chunk_data = chunked_data[i].copy()
                    chunk_data['embedding'] = embedding.embedding
                    chunk_data['embedding_metadata'] = {
                        'model_name': embedding.model_name,
                        'generation_timestamp': embedding.generation_timestamp
                    }
                    embeddings_data.append(chunk_data)
            
        except Exception as e:
            self.error_handler.add_error(f"Error generating embeddings: {str(e)}", "embedding_generation")
        
        return embeddings_data
    
    def _execute_database_upload(self, embeddings_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database upload step"""
        upload_results = {
            'uploaded_count': 0,
            'failed_count': 0,
            'success_rate': 0.0
        }
        
        try:
            # Initialize Qdrant manager
            self.qdrant_manager = QdrantManager(host="localhost", port=6333)
            
            if not self.qdrant_manager.health_check():
                raise Exception("Vector database is not available")
            
            collection_name = "tekyz_knowledge"
            
            # Prepare vectors for upload
            vectors = []
            for item in embeddings_data:
                vectors.append({
                    'id': item['chunk_id'],
                    'vector': item['embedding'],
                    'payload': {
                        'source': item['source'],
                        'content': item['content'],
                        'metadata': item['metadata']
                    }
                })
            
            # Upload vectors
            upload_result = self.qdrant_manager.upload_vectors(
                collection_name=collection_name,
                vectors=vectors,
                batch_size=config.get('batch_size', 100),
                progress_callback=lambda current, total: self.progress_tracker.update_step(
                    "database_upload", current, total
                )
            )
            
            upload_results = {
                'uploaded_count': upload_result.uploaded_count,
                'failed_count': upload_result.failed_count,
                'success_rate': upload_result.success_rate,
                'collection_name': collection_name
            }
            
        except Exception as e:
            self.error_handler.add_error(f"Error uploading to database: {str(e)}", "database_upload")
        
        return upload_results
    
    def _execute_validation(self, upload_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation step"""
        validation_results = {
            'validation_passed': False,
            'issues': []
        }
        
        try:
            if self.qdrant_manager and upload_results.get('uploaded_count', 0) > 0:
                # Basic validation: check if vectors were uploaded
                collection_name = upload_results.get('collection_name', 'tekyz_knowledge')
                collection_info = self.qdrant_manager.get_collection_info(collection_name)
                
                if collection_info and collection_info.points_count > 0:
                    validation_results['validation_passed'] = True
                    validation_results['points_count'] = collection_info.points_count
                else:
                    validation_results['issues'].append("No vectors found in database")
            else:
                validation_results['issues'].append("No data was uploaded to validate")
            
            self.progress_tracker.update_step("validation", 1, 1)
            
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {str(e)}")
            self.error_handler.add_error(f"Error during validation: {str(e)}", "validation")
        
        return validation_results 