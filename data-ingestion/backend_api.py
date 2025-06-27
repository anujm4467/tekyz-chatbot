#!/usr/bin/env python3
"""
FastAPI backend for the Tekyz Data Pipeline frontend
Provides REST API endpoints and WebSocket support for real-time updates
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from fastapi.staticfiles import StaticFiles

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

# from main import DataIngestionPipeline  # Temporarily disabled to avoid file handle issues

# Optional imports for file processing
try:
    import PyPDF2
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

try:
    import docx
    HAS_DOCX_SUPPORT = True
except ImportError:
    HAS_DOCX_SUPPORT = False

from src.database.pipeline_tracker import PipelineTracker, PipelineMetrics, PipelineJob
pipeline_tracker = PipelineTracker()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Tekyz Data Pipeline API",
    description="Backend API for the Tekyz data ingestion pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PipelineStatus(BaseModel):
    is_running: bool = False
    progress: int = 0
    current_step: str = ""
    logs: List[str] = []
    errors: List[Any] = []
    job_id: Optional[str] = None

# Global state management (WebSocketManager initialized after class definition)
active_jobs: Dict[str, PipelineJob] = {}
pipeline_tracker = PipelineTracker()

class PipelineJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = PipelineStatus(job_id=job_id)
        self.thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.metrics = PipelineMetrics(start_time=time.time())
        
    def start(self, file_data: List[dict], urls: List[str], websocket_manager):
        """Start the pipeline in a separate thread"""
        # Initialize tracking
        input_urls = urls if urls else []
        pipeline_tracker.start_job(self.job_id, input_urls)
        pipeline_tracker.start_step(self.job_id, "initialization")
        
        # Start processing thread
        self.thread = threading.Thread(
            target=self._run_pipeline,
            args=(file_data, urls, websocket_manager),
            daemon=True
        )
        self.thread.start()
        
    def stop(self):
        """Stop the pipeline"""
        self.should_stop.set()
        pipeline_tracker.add_log(self.job_id, "Pipeline stopped by user", 'WARNING')
        
    def _update_metrics(self, **kwargs):
        """Update pipeline metrics"""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        
        # Calculate ETA
        if self.metrics.processing_rate > 0 and self.metrics.total_urls > 0:
            remaining = self.metrics.total_urls - self.metrics.processed_urls
            if remaining > 0:
                self.metrics.estimated_completion = time.time() + (remaining / self.metrics.processing_rate)
        
        # Update tracker
        pipeline_tracker.update_metrics(self.job_id, self.metrics)
        
    def _run_pipeline(self, file_data: List[dict], urls: List[str], websocket_manager):
        """Execute the pipeline with enhanced tracking"""
        try:
            self.status.is_running = True
            self.status.current_step = "Initializing"
            self.status.progress = 0
            
            # Initialize metrics
            self.metrics.total_urls = len(urls) if urls else 0
            self._update_metrics()
            
            pipeline_tracker.add_log(self.job_id, "Pipeline execution started")
            pipeline_tracker.complete_step(self.job_id, "initialization", success=True)
            websocket_manager.broadcast(self._get_enhanced_status())
            
            # Save uploaded files
            if file_data:
                pipeline_tracker.start_step(self.job_id, "file_processing", len(file_data))
                self._save_file_data(file_data, websocket_manager)
                pipeline_tracker.complete_step(self.job_id, "file_processing", success=True)
                pipeline_tracker.add_log(self.job_id, f"Saved {len(file_data)} uploaded files")
            else:
                pipeline_tracker.skip_step(self.job_id, "file_processing", "No files to process")
                
            # Process URLs
            if urls:
                pipeline_tracker.start_step(self.job_id, "url_discovery", len(urls))
                self._process_urls(urls, websocket_manager)
                pipeline_tracker.complete_step(self.job_id, "url_discovery", success=True)
                pipeline_tracker.add_log(self.job_id, f"Processed {len(urls)} URLs for scraping")
            else:
                pipeline_tracker.skip_step(self.job_id, "url_discovery", "No URLs to process")
                
            # Run the standalone pipeline
            pipeline_tracker.add_log(self.job_id, "Starting content processing...")
            self._execute_pipeline_direct(websocket_manager)
            
            # Mark as completed
            pipeline_tracker.start_step(self.job_id, "finalization")
            pipeline_tracker.complete_step(self.job_id, "finalization", success=True)
            pipeline_tracker.complete_job(self.job_id, success=True)
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            self.status.errors.append({"message": str(e), "timestamp": datetime.now().isoformat()})
            
            # Complete current step as failed
            current_step = pipeline_tracker.get_current_step(self.job_id)
            if current_step:
                pipeline_tracker.complete_step(self.job_id, current_step.step_name, success=False, error_message=error_msg)
            
            pipeline_tracker.complete_job(self.job_id, success=False, error_message=error_msg)
            websocket_manager.broadcast(self._get_enhanced_status())
        finally:
            self.status.is_running = False
            self.status.current_step = "Completed"
            self.metrics.end_time = time.time()
            self._update_metrics()
            websocket_manager.broadcast(self._get_enhanced_status())
    
    def _get_enhanced_status(self) -> dict:
        """Get enhanced status with ETA and metrics"""
        status_dict = self.status.dict()
        
        # Add metrics and ETA
        status_dict.update({
            'metrics': {
                'total_urls': self.metrics.total_urls,
                'processed_urls': self.metrics.processed_urls,
                'successful_urls': self.metrics.successful_urls,
                'failed_urls': self.metrics.failed_urls,
                'total_chunks': self.metrics.total_chunks,
                'total_embeddings': self.metrics.total_embeddings,
                'total_vectors': self.metrics.total_vectors,
                'duplicate_count': self.metrics.duplicate_count,
                'processing_rate': round(self.metrics.processing_rate, 2),
                'elapsed_time': round(self.metrics.elapsed_time, 1),
                'progress_percentage': round(self.metrics.progress_percentage, 1)
            },
            'eta': {
                'eta_seconds': self.metrics.eta_seconds,
                'eta_formatted': self._format_eta(self.metrics.eta_seconds) if self.metrics.eta_seconds else None,
                'estimated_completion': self.metrics.estimated_completion
            }
        })
        
        return status_dict
    
    def _format_eta(self, eta_seconds: float) -> str:
        """Format ETA in human-readable format"""
        if eta_seconds <= 0:
            return "Calculating..."
        
        if eta_seconds < 60:
            return f"{int(eta_seconds)} seconds"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds / 60)
            seconds = int(eta_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    async def _save_uploaded_files(self, files: List[UploadFile], websocket_manager):
        """Save uploaded files to the data/raw directory"""
        self.status.current_step = "Saving uploaded files"
        self.status.progress = 10
        websocket_manager.broadcast(self.status.dict())
        
        upload_dir = Path("data/raw/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        for i, file in enumerate(files):
            if self.should_stop.is_set():
                return
                
            try:
                file_path = upload_dir / file.filename
                self.status.logs.append(f"Saving file: {file.filename}")
                websocket_manager.broadcast(self.status.dict())
                
                # Use async read for FastAPI UploadFile
                content = await file.read()
                
                # Save to disk
                with open(file_path, "wb") as f:
                    f.write(content)
                    
                log_msg = f"✅ Saved file: {file.filename} ({len(content)} bytes)"
                self.status.logs.append(log_msg)
                logger.info(log_msg)
                
            except Exception as e:
                error_msg = f"❌ Failed to save {file.filename}: {str(e)}"
                self.status.logs.append(error_msg)
                self.status.errors.append({"message": error_msg, "timestamp": datetime.now().isoformat()})
                logger.error(error_msg)
                websocket_manager.broadcast(self.status.dict())
                continue
            
            progress = 10 + (i + 1) / len(files) * 10
            self.status.progress = int(progress)
            websocket_manager.broadcast(self.status.dict())
    
    def _save_file_data(self, file_data: List[dict], websocket_manager):
        """Save pre-read file data to the data/raw directory"""
        self.status.current_step = "Saving uploaded files"
        self.status.progress = 10
        websocket_manager.broadcast(self.status.dict())
        
        upload_dir = Path("data/raw/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        for i, file_info in enumerate(file_data):
            if self.should_stop.is_set():
                return
                
            try:
                filename = file_info['filename']
                content = file_info['content']
                
                file_path = upload_dir / filename
                self.status.logs.append(f"Saving file: {filename}")
                websocket_manager.broadcast(self.status.dict())
                
                # Save to disk
                with open(file_path, "wb") as f:
                    f.write(content)
                    
                log_msg = f"✅ Saved file: {filename} ({len(content)} bytes)"
                self.status.logs.append(log_msg)
                logger.info(log_msg)
                
            except Exception as e:
                error_msg = f"❌ Failed to save {file_info.get('filename', 'unknown')}: {str(e)}"
                self.status.logs.append(error_msg)
                self.status.errors.append({"message": error_msg, "timestamp": datetime.now().isoformat()})
                logger.error(error_msg)
                websocket_manager.broadcast(self.status.dict())
                continue
            
            progress = 10 + (i + 1) / len(file_data) * 10
            self.status.progress = int(progress)
            websocket_manager.broadcast(self.status.dict())
            
    def _process_urls(self, urls: List[str], websocket_manager):
        """Process URL list for scraping"""
        self.status.current_step = "Processing URLs"
        self.status.progress = 20
        websocket_manager.broadcast(self.status.dict())
        
        # Save URLs to a file for the pipeline to process
        # Make sure we don't overwrite existing URLs, append instead
        urls_file = Path("data/raw/urls.txt")
        
        # Read existing URLs if file exists
        existing_urls = set()
        if urls_file.exists():
            with open(urls_file, "r") as f:
                existing_urls = {line.strip() for line in f if line.strip()}
        
        # Add new URLs that aren't already present
        new_urls = [url for url in urls if url not in existing_urls]
        
        if new_urls:
            with open(urls_file, "a") as f:  # Append mode
                for url in new_urls:
                    f.write(f"{url}\n")
            log_msg = f"Added {len(new_urls)} new URLs for processing (total: {len(existing_urls) + len(new_urls)})"
        else:
            log_msg = f"All {len(urls)} URLs already exist in processing queue"
            
        self.status.logs.append(log_msg)
        logger.info(log_msg)
        
    def _execute_pipeline_direct(self, websocket_manager):
        """Execute the pipeline directly without the problematic DataIngestionPipeline class"""
        try:
            self.status.logs.append("Starting direct pipeline execution...")
            websocket_manager.broadcast(self.status.dict())
            
            # Run the simplified pipeline without any complex dependencies
            result = asyncio.run(self._run_standalone_pipeline(websocket_manager))
            
            if result.get('success', False):
                self.status.progress = 100
                self.status.logs.append("Pipeline completed successfully!")
                
                # Handle results
                if 'processed_chunks' in result:
                    self.status.logs.append(f"✅ Processed {result['processed_chunks']} text chunks")
                    
                websocket_manager.broadcast(self.status.dict())
            else:
                error_msg = result.get('error', 'Pipeline failed with unknown error')
                self.status.errors.append({"message": error_msg, "timestamp": datetime.now().isoformat()})
                self.status.logs.append(f"❌ Pipeline failed: {error_msg}")
                websocket_manager.broadcast(self.status.dict())
                
        except Exception as e:
            logger.error(f"Direct pipeline execution failed: {str(e)}")
            self.status.errors.append({"message": str(e), "timestamp": datetime.now().isoformat()})
            self.status.logs.append(f"❌ Pipeline failed: {str(e)}")
            websocket_manager.broadcast(self.status.dict())
            raise
        
    def _execute_pipeline(self, pipeline, websocket_manager):
        """Execute the main data pipeline"""
        try:
            self.status.logs.append("Initializing pipeline execution...")
            websocket_manager.broadcast(self.status.dict())
            
            # Run a simplified pipeline to avoid file handle issues
            result = self._run_simple_pipeline(pipeline, websocket_manager)
            
            if result.get('success', False):
                self.status.progress = 100
                self.status.logs.append("Pipeline completed successfully!")
                
                # Handle simplified pipeline results
                if 'processed_chunks' in result:
                    self.status.logs.append(f"✅ Processed {result['processed_chunks']} text chunks")
                
                if 'generated_embeddings' in result:
                    self.status.logs.append(f"✅ Generated {result['generated_embeddings']} embeddings")
                
                if 'uploaded_vectors' in result:
                    self.status.logs.append(f"✅ Uploaded {result['uploaded_vectors']} vectors to database")
                    
                # Handle complex pipeline results (legacy format)
                if 'scraping' in result:
                    self.status.logs.append(f"✅ Scraping: {result.get('scraping', {}).get('scraped_pages', 0)} pages")
                
                if 'processing' in result:
                    processing = result['processing']['result']
                    self.status.logs.append(f"✅ Processing: {processing.total_chunks} chunks created")
                
                if 'embeddings' in result:
                    embeddings = result['embeddings']['result']
                    self.status.logs.append(f"✅ Embeddings: {embeddings.successful_embeddings} vectors generated")
                
                if 'database' in result:
                    database = result['database']
                    self.status.logs.append(f"✅ Database: {database.get('uploaded_vectors', 0)} vectors uploaded")
                    
                websocket_manager.broadcast(self.status.dict())
            else:
                error_msg = result.get('error', 'Pipeline failed with unknown error')
                self.status.errors.append({"message": error_msg, "timestamp": datetime.now().isoformat()})
                self.status.logs.append(f"❌ Pipeline failed: {error_msg}")
                websocket_manager.broadcast(self.status.dict())
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            self.status.errors.append({"message": str(e), "timestamp": datetime.now().isoformat()})
            self.status.logs.append(f"❌ Pipeline failed: {str(e)}")
            websocket_manager.broadcast(self.status.dict())
            raise
            
    async def _run_standalone_pipeline(self, websocket_manager):
        """Run a complete pipeline with text processing, embedding generation, and vector storage"""
        try:
            processed_chunks = 0
            generated_embeddings = 0
            uploaded_vectors = 0
            
            # Check for URLs first
            urls_file = Path("data/raw/urls.txt")
            has_urls = urls_file.exists() and urls_file.stat().st_size > 0
            
            # Check for uploaded files
            upload_dir = Path("data/raw/uploads")
            has_uploads = upload_dir.exists() and list(upload_dir.glob("*"))
            
            if has_urls:
                # Read URLs from file first
                with open(urls_file, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]
                
                pipeline_tracker.start_step(self.job_id, "web_scraping", len(urls))
                self.status.logs.append(f"Found {len(urls)} URLs to scrape")
                pipeline_tracker.update_step_progress(self.job_id, "web_scraping", 0, 5)
                websocket_manager.broadcast(self.status.dict())
                
                # Step 1: Web Scraping
                self.status.current_step = "Web Scraping"
                self.status.progress = 10
                websocket_manager.broadcast(self.status.dict())
                
                scraped_data = await self._scrape_urls(urls, websocket_manager)
                
                # Improved success/failure logic for web scraping
                if scraped_data and len(scraped_data) > 0:
                    # Calculate success rate
                    total_attempted = len(urls)  # This will be updated by recursive crawling
                    successful_scraped = len(scraped_data)
                    
                    # Get actual metrics from the scraping process
                    if hasattr(self, 'metrics') and hasattr(self.metrics, 'total_urls'):
                        total_attempted = max(self.metrics.total_urls, len(urls))
                    
                    success_rate = (successful_scraped / total_attempted) * 100 if total_attempted > 0 else 0
                    
                    if successful_scraped == total_attempted:
                        # Complete success
                        pipeline_tracker.complete_step(self.job_id, "web_scraping", success=True)
                        self.status.logs.append(f"✅ Successfully scraped all {successful_scraped} pages")
                    elif successful_scraped > 0:
                        # Partial success - mark as completed but with warning
                        warning_msg = f"Partial success: {successful_scraped}/{total_attempted} pages scraped ({success_rate:.1f}% success rate)"
                        pipeline_tracker.complete_step(self.job_id, "web_scraping", success=True, 
                                                     error_message=warning_msg)
                        self.status.logs.append(f"⚠️ {warning_msg}")
                    else:
                        # This case should not happen if scraped_data has items, but keep as safeguard
                        pipeline_tracker.complete_step(self.job_id, "web_scraping", success=False, 
                                                     error_message="No content successfully scraped")
                        self.status.logs.append("❌ No content successfully scraped from URLs")
                    
                    # Continue with processing the scraped data
                    # Step 2: Process scraped content
                    pipeline_tracker.start_step(self.job_id, "content_cleaning", len(scraped_data))
                    self.status.current_step = "Processing Scraped Content"
                    self.status.progress = 30
                    websocket_manager.broadcast(self.status.dict())
                    
                    chunks = self._process_with_real_pipeline(scraped_data, websocket_manager)
                    processed_chunks += len(chunks)
                    pipeline_tracker.complete_step(self.job_id, "content_cleaning", success=True)
                    
                    # Step 3: Generate embeddings
                    pipeline_tracker.start_step(self.job_id, "embedding_generation", len(chunks))
                    
                    self.status.current_step = "Generating Embeddings"
                    self.status.progress = 80
                    self.status.logs.append("Generating vector embeddings...")
                    websocket_manager.broadcast(self.status.dict())
                    
                    embeddings_data = self._generate_embeddings(chunks, websocket_manager)
                    generated_embeddings += len(embeddings_data)
                    
                    pipeline_tracker.complete_step(self.job_id, "embedding_generation", success=True)
                    
                    self.status.logs.append(f"Generated embeddings for {len(embeddings_data)} chunks")
                    websocket_manager.broadcast(self.status.dict())
                    
                    # Step 4: Upload to vector database with deduplication
                    pipeline_tracker.start_step(self.job_id, "database_upload", len(embeddings_data))
                    
                    self.status.current_step = "Storing Vectors"
                    self.status.progress = 90
                    self.status.logs.append("Uploading vectors to database with deduplication...")
                    websocket_manager.broadcast(self.status.dict())
                    
                    upload_results = self._upload_to_vector_db_with_dedup(embeddings_data, websocket_manager)
                    uploaded_vectors += upload_results.get('uploaded_count', 0)
                    
                    pipeline_tracker.complete_step(self.job_id, "database_upload", success=True)
                else:
                    # Complete failure - no pages scraped successfully
                    total_attempted = len(urls)
                    if hasattr(self, 'metrics') and hasattr(self.metrics, 'total_urls'):
                        total_attempted = max(self.metrics.total_urls, len(urls))
                    
                    pipeline_tracker.complete_step(self.job_id, "web_scraping", success=False, 
                                                 error_message=f"Failed to scrape any content from {total_attempted} URLs")
                    self.status.logs.append(f"❌ Failed to scrape any content from {total_attempted} URLs")
                    
                    # Skip web scraping related steps since no content was scraped
                    if has_uploads:
                        # We have uploads to process, so skip web-scraping-specific steps
                        pipeline_tracker.skip_step(self.job_id, "content_cleaning", "No web content scraped, using uploaded files")
                    else:
                        # No uploads either, skip content processing steps
                        pipeline_tracker.skip_step(self.job_id, "content_cleaning", "No content to clean")
                        pipeline_tracker.skip_step(self.job_id, "text_chunking", "No content to chunk")
                        pipeline_tracker.skip_step(self.job_id, "embedding_generation", "No content to embed")
                        pipeline_tracker.skip_step(self.job_id, "deduplication", "No content to deduplicate")
                        pipeline_tracker.skip_step(self.job_id, "database_upload", "No content to upload")
            
            if has_uploads:
                if not has_urls:  # Only start if not already started for URLs
                    pipeline_tracker.start_step(self.job_id, "content_cleaning")
                
                self.status.logs.append("Processing uploaded files...")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 1: Create scraped data from uploads
                self.status.current_step = "Processing Uploads"
                self.status.progress = 50
                scraped_data_file = self._create_scraped_data_from_uploads(upload_dir)
                self.status.logs.append(f"Created data file: {scraped_data_file}")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 2: Advanced text processing and chunking
                pipeline_tracker.start_step(self.job_id, "text_chunking")
                self.status.current_step = "Text Processing & Chunking"
                self.status.progress = 60
                websocket_manager.broadcast(self.status.dict())
                
                # Load the scraped data
                with open(scraped_data_file, 'r', encoding='utf-8') as f:
                    scraped_data = json.load(f)
                
                # Use the real text processor
                all_chunks = self._process_with_real_pipeline(scraped_data, websocket_manager)
                processed_chunks += len(all_chunks)
                pipeline_tracker.complete_step(self.job_id, "text_chunking", success=True)
                
                self.status.logs.append(f"Created {len(all_chunks)} text chunks")
                self.status.progress = 70
                websocket_manager.broadcast(self.status.dict())
                
                # Step 3: Generate embeddings
                pipeline_tracker.start_step(self.job_id, "embedding_generation", len(all_chunks))
                
                self.status.current_step = "Generating Embeddings"
                self.status.progress = 80
                self.status.logs.append("Generating vector embeddings...")
                websocket_manager.broadcast(self.status.dict())
                
                embeddings_data = self._generate_embeddings(all_chunks, websocket_manager)
                generated_embeddings += len(embeddings_data)
                
                pipeline_tracker.complete_step(self.job_id, "embedding_generation", success=True)
                
                self.status.logs.append(f"Generated embeddings for {len(embeddings_data)} chunks")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 4: Upload to vector database with deduplication
                pipeline_tracker.start_step(self.job_id, "database_upload", len(embeddings_data))
                
                self.status.current_step = "Storing Vectors"
                self.status.progress = 90
                self.status.logs.append("Uploading vectors to database with deduplication...")
                websocket_manager.broadcast(self.status.dict())
                
                upload_results = self._upload_to_vector_db_with_dedup(embeddings_data, websocket_manager)
                uploaded_vectors += upload_results.get('uploaded_count', 0)
                
                pipeline_tracker.complete_step(self.job_id, "database_upload", success=True)
            
            if not has_urls and not has_uploads:
                # Skip all content processing steps when no data
                pipeline_tracker.skip_step(self.job_id, "web_scraping", "No URLs provided")
                pipeline_tracker.skip_step(self.job_id, "content_cleaning", "No content to clean")
                pipeline_tracker.skip_step(self.job_id, "text_chunking", "No content to chunk")
                pipeline_tracker.skip_step(self.job_id, "embedding_generation", "No content to embed")
                pipeline_tracker.skip_step(self.job_id, "deduplication", "No content to deduplicate")
                pipeline_tracker.skip_step(self.job_id, "database_upload", "No content to upload")
                
                # Still complete finalization step
                pipeline_tracker.start_step(self.job_id, "finalization")
                pipeline_tracker.complete_step(self.job_id, "finalization", success=False, 
                                             error_message="No data to process - no URLs or uploaded files found")
                pipeline_tracker.complete_job(self.job_id, success=False, 
                                            error_message="No data to process - no URLs or uploaded files found")
                
                return {'success': False, 'error': 'No data to process - no URLs or uploaded files found'}
            
            # Skip unused steps based on what data we have
            if not has_uploads:
                # Skip file processing step when only URLs provided
                pipeline_tracker.skip_step(self.job_id, "file_processing", "No files uploaded")
            
            # Skip remaining steps if not used and finalize job
            if not has_urls:
                pipeline_tracker.skip_step(self.job_id, "web_scraping", "No URLs provided")
            if not has_uploads and not has_urls:
                pipeline_tracker.skip_step(self.job_id, "content_cleaning", "No content to clean")
                pipeline_tracker.skip_step(self.job_id, "text_chunking", "No content to chunk")
                pipeline_tracker.skip_step(self.job_id, "embedding_generation", "No content to embed")
                pipeline_tracker.skip_step(self.job_id, "deduplication", "No content to deduplicate")
                pipeline_tracker.skip_step(self.job_id, "database_upload", "No content to upload")
            
            # Mark finalization step
            pipeline_tracker.start_step(self.job_id, "finalization")
            
            self.status.logs.append(f"✅ Pipeline completed: {processed_chunks} chunks, {generated_embeddings} embeddings, {uploaded_vectors} new vectors")
            self.status.progress = 100
            self.status.current_step = "Complete"
            websocket_manager.broadcast(self.status.dict())
            
            # Complete finalization and job
            pipeline_tracker.complete_step(self.job_id, "finalization", success=True)
            pipeline_tracker.complete_job(self.job_id, success=True)
            
            return {
                'success': True,
                'processed_chunks': processed_chunks,
                'generated_embeddings': generated_embeddings,
                'uploaded_vectors': uploaded_vectors,
                'pipeline_completed_at': datetime.now().isoformat()
            }
                    
        except Exception as e:
            logger.error(f"Standalone pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    async def _scrape_urls(self, urls: List[str], websocket_manager):
        """Scrape content from URLs using recursive crawling with dynamic link discovery"""
        try:
            from src.scraper.orchestrator import ScrapingOrchestrator
            from src.scraper.url_discovery import URLDiscovery
            
            # Initialize components
            orchestrator = ScrapingOrchestrator()
            url_discovery = URLDiscovery()
            
            # Track discovered URLs and already processed ones
            discovered_urls = set(urls)  # Start with user-provided URLs
            processed_urls = set()
            scraped_pages = []
            
            # Determine base domain for crawling scope
            base_domain = None
            if urls:
                from urllib.parse import urlparse
                base_domain = urlparse(urls[0]).netloc
                pipeline_tracker.add_log(self.job_id, f"Starting recursive crawling for domain: {base_domain}")
            else:
                pipeline_tracker.add_log(self.job_id, "No URLs provided for crawling", 'WARNING')
                return []
            
            websocket_manager.broadcast(self._get_enhanced_status())
            
            # Recursive crawling loop
            crawl_depth = 0
            max_depth = 5  # Prevent infinite crawling
            
            while discovered_urls and crawl_depth < max_depth:
                crawl_depth += 1
                
                # Update metrics
                self.metrics.total_urls = len(discovered_urls)
                self._update_metrics()
                
                pipeline_tracker.add_log(self.job_id, f"Crawl depth {crawl_depth}: {len(discovered_urls)} URLs to process")
                websocket_manager.broadcast(self._get_enhanced_status())
                
                # Get URLs to process in this iteration
                current_batch = list(discovered_urls - processed_urls)
                
                if not current_batch:
                    break
                
                # Limit batch size to prevent overwhelming
                batch_size = min(20, len(current_batch))
                current_batch = current_batch[:batch_size]
                
                for i, url in enumerate(current_batch):
                    if self.should_stop.is_set():
                        break
                    
                    # Skip if already processed
                    if url in processed_urls:
                        continue
                    
                    pipeline_tracker.add_log(self.job_id, f"Scraping ({i+1}/{len(current_batch)}): {url}")
                    websocket_manager.broadcast(self._get_enhanced_status())
                    
                    try:
                        # Scrape the page
                        page_data = await orchestrator.scrape_single_page(url)
                        
                        if page_data and (page_data.get('content') or page_data.get('text')):
                            scraped_pages.append(page_data)
                            processed_urls.add(url)
                            
                            # Update success metrics
                            self.metrics.processed_urls = len(processed_urls)
                            self.metrics.successful_urls = len(scraped_pages)
                            self._update_metrics()
                            
                            # Extract new links from this page
                            page_content = page_data.get('content', '')
                            new_urls = url_discovery.discover_urls_from_page(page_content, url)
                            
                            # Filter new URLs to same domain and validate
                            valid_new_urls = []
                            for new_url in new_urls:
                                try:
                                    from urllib.parse import urlparse
                                    new_domain = urlparse(new_url).netloc
                                    
                                    # Only add URLs from the same domain
                                    if (new_domain == base_domain and 
                                        new_url not in processed_urls and 
                                        new_url not in discovered_urls and
                                        url_discovery.validate_url(new_url, base_domain)):
                                        
                                        valid_new_urls.append(new_url)
                                        
                                except Exception:
                                    continue
                            
                            # Add new URLs to discovery queue
                            if valid_new_urls:
                                discovered_urls.update(valid_new_urls)
                                self.metrics.total_urls = len(discovered_urls)
                                pipeline_tracker.add_log(self.job_id, f"Found {len(valid_new_urls)} new links on {url}")
                            
                            pipeline_tracker.add_log(self.job_id, f"✅ Successfully scraped: {url}")
                            
                        else:
                            processed_urls.add(url)  # Mark as processed even if failed
                            self.metrics.processed_urls = len(processed_urls)
                            self.metrics.failed_urls = self.metrics.processed_urls - self.metrics.successful_urls
                            self._update_metrics()
                            pipeline_tracker.add_log(self.job_id, f"❌ Failed to scrape: {url}")
                            
                    except Exception as e:
                        processed_urls.add(url)
                        self.metrics.processed_urls = len(processed_urls)
                        self.metrics.failed_urls = self.metrics.processed_urls - self.metrics.successful_urls
                        self._update_metrics()
                        pipeline_tracker.add_log(self.job_id, f"❌ Error scraping {url}: {str(e)}")
                    
                    # Update progress
                    total_discovered = len(discovered_urls)
                    total_processed = len(processed_urls)
                    progress = 10 + (total_processed / max(total_discovered, 1)) * 15  # 10-25% for scraping
                    self.status.progress = int(progress)
                    websocket_manager.broadcast(self._get_enhanced_status())
                
                # Log progress after each depth level
                total_discovered = len(discovered_urls)
                total_processed = len(processed_urls)
                remaining = total_discovered - total_processed
                
                pipeline_tracker.add_log(self.job_id, f"Depth {crawl_depth} complete: {total_processed} processed, {remaining} remaining, {len(scraped_pages)} successful")
                websocket_manager.broadcast(self._get_enhanced_status())
                
                # Break if no new URLs were discovered
                if remaining == 0:
                    break
            
            # Final summary
            total_discovered = len(discovered_urls)
            total_processed = len(processed_urls)
            
            pipeline_tracker.add_log(self.job_id, f"🎯 Recursive crawling completed!")
            pipeline_tracker.add_log(self.job_id, f"📈 Total discovered URLs: {total_discovered}")
            pipeline_tracker.add_log(self.job_id, f"✅ Successfully scraped: {len(scraped_pages)}")
            pipeline_tracker.add_log(self.job_id, f"🚫 Failed/skipped: {total_processed - len(scraped_pages)}")
            pipeline_tracker.add_log(self.job_id, f"🔄 Maximum crawl depth: {crawl_depth}")
            
            websocket_manager.broadcast(self._get_enhanced_status())
            
            return scraped_pages
            
        except ImportError as e:
            pipeline_tracker.add_log(self.job_id, f"❌ Scraping components not available: {e}", 'ERROR')
            return []
        except Exception as e:
            pipeline_tracker.add_log(self.job_id, f"❌ Recursive crawling failed: {str(e)}", 'ERROR')
            return []
    
    def _simple_text_chunking(self, text: str, url: str):
        """Simple text chunking without external dependencies"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 50:  # Only process meaningful paragraphs
                # Split long paragraphs by sentences (simple approach)
                sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
                
                # Group sentences into chunks of ~500 characters
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) < 500:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'url': url,
                                'chunk_id': len(chunks)
                            })
                        current_chunk = sentence + ". "
                
                # Add remaining chunk
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'url': url,
                        'chunk_id': len(chunks)
                    })
        
        return chunks
    
    def _process_with_real_pipeline(self, scraped_data, websocket_manager):
        """Process scraped data using the real text processor"""
        try:
            from src.processor.text_processor import TextProcessor
            from src.processor.text_processor import ChunkingConfig
            
            # Create chunking configuration
            chunking_config = ChunkingConfig(
                max_chunk_size=1000,
                min_chunk_size=100,
                overlap_size=100,
                split_by_sentences=True,
                preserve_paragraphs=True,
                respect_word_boundaries=True
            )
            
            # Initialize text processor
            text_processor = TextProcessor(chunking_config)
            
            # Handle both list and dict formats
            if isinstance(scraped_data, list):
                pages = scraped_data
            else:
                pages = scraped_data.get('pages', [])
            
            # Process each page
            all_chunks = []
            for page in pages:
                try:
                    chunks = text_processor.process_page_content(page)
                    all_chunks.extend(chunks)
                except Exception as e:
                    page_url = page.get('url', 'unknown') if isinstance(page, dict) else 'unknown'
                    logger.error(f"Failed to process page {page_url}: {e}")
                    continue
            
            return all_chunks
            
        except ImportError as e:
            logger.error(f"Failed to import text processor: {e}")
            # Fallback to simple chunking
            return self._simple_fallback_chunking(scraped_data)
        except Exception as e:
            error_msg = f"Text processing failed: {e}"
            logger.error(error_msg)
            pipeline_tracker.add_log(self.job_id, error_msg, 'ERROR')
            # Fallback to simple chunking
            return self._simple_fallback_chunking(scraped_data)
    
    def _simple_fallback_chunking(self, scraped_data):
        """Fallback simple chunking if real processor fails"""
        all_chunks = []
        
        # Handle both list and dict formats
        if isinstance(scraped_data, list):
            pages = scraped_data
        else:
            pages = scraped_data.get('pages', [])
        
        for page in pages:
            # Handle different page formats
            if isinstance(page, dict):
                content = page.get('text', '') or page.get('content', '')
                url = page.get('url', '')
            else:
                content = str(page)
                url = ''
                
            if content and content.strip():
                chunks = self._simple_text_chunking(content, url)
                all_chunks.extend(chunks)
        return all_chunks
    
    def _generate_embeddings(self, chunks, websocket_manager):
        """Generate embeddings for text chunks"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load embedding model
            self.status.logs.append("Loading embedding model...")
            websocket_manager.broadcast(self.status.dict())
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Prepare texts for embedding
            texts = []
            for chunk in chunks:
                if hasattr(chunk, 'clean_text'):
                    texts.append(chunk.clean_text)
                elif hasattr(chunk, 'text'):
                    texts.append(chunk.text)
                else:
                    texts.append(str(chunk))
            
            self.status.logs.append(f"Generating embeddings for {len(texts)} text chunks...")
            websocket_manager.broadcast(self.status.dict())
            
            # Generate embeddings in batches
            batch_size = 32
            embeddings_data = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_chunks = chunks[i:i+batch_size]
                
                # Generate embeddings
                embeddings = model.encode(batch_texts, normalize_embeddings=True)
                
                # Create embedding data
                for j, (text, chunk, embedding) in enumerate(zip(batch_texts, batch_chunks, embeddings)):
                    embedding_data = {
                        'chunk_id': getattr(chunk, 'id', f'chunk_{i+j}'),
                        'text': text,
                        'embedding': embedding.tolist(),
                        'metadata': {
                            'source_url': getattr(chunk, 'source_url', ''),
                            'chunk_index': getattr(chunk, 'chunk_index', i+j),
                            'char_count': getattr(chunk, 'char_count', len(text)),
                            'word_count': getattr(chunk, 'word_count', len(text.split())),
                            'model_name': 'all-MiniLM-L6-v2',
                            'embedding_dim': len(embedding)
                        }
                    }
                    embeddings_data.append(embedding_data)
                
                # Update progress
                progress = 70 + (i + len(batch_texts)) / len(texts) * 15  # 70-85% range
                self.status.progress = int(progress)
                websocket_manager.broadcast(self.status.dict())
            
            return embeddings_data
            
        except ImportError as e:
            logger.error(f"SentenceTransformers not available: {e}")
            raise Exception("Embedding generation requires sentence-transformers package")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _upload_to_vector_db_with_dedup(self, embeddings_data, websocket_manager):
        """Upload embeddings to Qdrant vector database with deduplication"""
        try:
            from src.database.qdrant_client import QdrantManager
            from src.database.models import VectorConfig
            
            # Initialize Qdrant client
            self.status.logs.append("Connecting to vector database...")
            websocket_manager.broadcast(self.status.dict())
            
            config = VectorConfig(
                host="localhost",
                port=6333,
                collection_name="tekyz_knowledge"
            )
            qdrant_manager = QdrantManager(config=config)
            
            # Connect to database
            if not qdrant_manager.connect():
                raise Exception("Vector database is not available. Please ensure Qdrant is running.")
            
            # Check health and log status
            health_status = qdrant_manager.health_check()
            self.status.logs.append(f"Health check: connected={health_status.get('connected')}, collection_exists={health_status.get('collection_exists')}")
            websocket_manager.broadcast(self.status.dict())
            
            if not health_status.get('connected', False):
                raise Exception(f"Vector database connection failed: {health_status.get('errors', [])}")
            
            # Log existing data if collection exists
            if health_status.get('collection_exists'):
                collection_info = health_status.get('collection_info', {})
                points_count = collection_info.get('points_count', 0)
                self.status.logs.append(f"Collection already exists with {points_count} vectors. Adding new vectors...")
            else:
                self.status.logs.append("Collection doesn't exist yet. Will create it...")
            
            websocket_manager.broadcast(self.status.dict())
            
            # Create collection if it doesn't exist (will skip if already exists)
            self.status.logs.append(f"Ensuring collection exists: {config.collection_name}")
            websocket_manager.broadcast(self.status.dict())
            
            if qdrant_manager.create_collection(recreate=False):
                self.status.logs.append("✅ Collection ready for vector upload")
            else:
                self.status.logs.append("⚠️ Collection setup had issues, but proceeding...")
            
            websocket_manager.broadcast(self.status.dict())
            
            # Deduplication: Get all existing content hashes in one batch operation
            pipeline_tracker.start_step(self.job_id, "deduplication", len(embeddings_data))
            self.status.logs.append("Checking for duplicate content (batch operation)...")
            websocket_manager.broadcast(self.status.dict())
            
            # Get all existing content hashes from the database in one operation
            existing_hashes = qdrant_manager.get_existing_content_hashes()
            
            new_embeddings = []
            duplicate_count = 0
            
            for i, embedding_data in enumerate(embeddings_data):
                # Create a hash of the text content for deduplication
                text_content = embedding_data.get('text', '')
                content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
                
                # Add content hash to metadata for future deduplication
                if 'metadata' not in embedding_data:
                    embedding_data['metadata'] = {}
                embedding_data['metadata']['content_hash'] = content_hash
                
                # Check if this content already exists using the batch-loaded hashes
                if content_hash not in existing_hashes:
                    new_embeddings.append(embedding_data)
                    # Add to existing hashes to avoid duplicates within this batch
                    existing_hashes.add(content_hash)
                else:
                    duplicate_count += 1
                    logger.info(f"Skipping duplicate content: {content_hash[:12]}...")
                
                # Update progress every 50 items (less frequent updates)
                if (i + 1) % 50 == 0:
                    pipeline_tracker.update_step_progress(self.job_id, "deduplication", i + 1)
                    self.status.logs.append(f"Processed {i + 1}/{len(embeddings_data)} chunks for deduplication")
                    websocket_manager.broadcast(self.status.dict())
            
            pipeline_tracker.complete_step(self.job_id, "deduplication", success=True)
            self.status.logs.append(f"Found {len(new_embeddings)} new chunks, {duplicate_count} duplicates skipped")
            websocket_manager.broadcast(self.status.dict())
            
            if not new_embeddings:
                self.status.logs.append("No new content to upload - all content already exists")
                return {
                    'uploaded_count': 0,
                    'failed_count': 0,
                    'duplicate_count': duplicate_count,
                    'success': True
                }
            
            # Upload vectors using the QdrantManager
            self.status.logs.append(f"Uploading {len(new_embeddings)} vectors to collection {config.collection_name}...")
            websocket_manager.broadcast(self.status.dict())
            
            upload_result = qdrant_manager.upload_vectors(
                embeddings=new_embeddings,
                progress_callback=lambda **kwargs: self.status.logs.append(f"Upload progress: {kwargs}")
            )
            
            return {
                'uploaded_count': upload_result.uploaded_vectors,
                'failed_count': upload_result.failed_vectors,
                'duplicate_count': duplicate_count,
                'success_rate': upload_result.success_rate,
                'collection_name': config.collection_name,
                'success': upload_result.success
            }
            
        except ImportError as e:
            logger.error(f"Database components not available: {e}")
            raise Exception("Vector database upload requires Qdrant components")
        except Exception as e:
            logger.error(f"Vector database upload failed: {e}")
            raise
        
    def _run_simple_pipeline(self, pipeline, websocket_manager):
        """Run a simplified synchronous pipeline"""
        try:
            # Check if we have uploaded files to process
            upload_dir = Path("data/raw/uploads")
            if upload_dir.exists() and list(upload_dir.glob("*")):
                self.status.logs.append("Found uploaded files, processing...")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 1: Create scraped data from uploads
                self.status.current_step = "Processing Uploads"
                self.status.progress = 20
                scraped_data_file = self._create_scraped_data_from_uploads(upload_dir)
                self.status.logs.append(f"Created data file: {scraped_data_file}")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 2: Run text processing
                self.status.current_step = "Text Processing"
                self.status.progress = 40
                self.status.logs.append("Processing text content...")
                websocket_manager.broadcast(self.status.dict())
                
                # Call the text processor directly
                from src.processor.text_processor import TextProcessor
                processor = TextProcessor()
                
                # Load the scraped data
                with open(scraped_data_file, 'r', encoding='utf-8') as f:
                    scraped_data = json.load(f)
                
                # Process each page
                all_chunks = []
                for page in scraped_data.get('pages', []):
                    content = page.get('content', '')
                    if content and content.strip():
                        chunks = processor.process_text(content, page.get('url', ''))
                        all_chunks.extend(chunks)
                
                self.status.logs.append(f"Created {len(all_chunks)} text chunks")
                self.status.progress = 60
                websocket_manager.broadcast(self.status.dict())
                
                # Step 3: Generate embeddings
                pipeline_tracker.start_step(self.job_id, "embedding_generation", len(all_chunks))
                
                self.status.current_step = "Generating Embeddings"
                self.status.progress = 80
                self.status.logs.append("Generating embeddings...")
                websocket_manager.broadcast(self.status.dict())
                
                embeddings_data = self._generate_embeddings(all_chunks, websocket_manager)
                generated_embeddings += len(embeddings_data)
                
                pipeline_tracker.complete_step(self.job_id, "embedding_generation", success=True)
                
                self.status.logs.append(f"Generated embeddings for {len(embeddings_data)} chunks")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 4: Upload to vector database with deduplication
                pipeline_tracker.start_step(self.job_id, "database_upload", len(embeddings_data))
                
                self.status.current_step = "Storing Vectors"
                self.status.progress = 90
                self.status.logs.append("Uploading vectors to database with deduplication...")
                websocket_manager.broadcast(self.status.dict())
                
                upload_results = self._upload_to_vector_db_with_dedup(embeddings_data, websocket_manager)
                uploaded_vectors += upload_results.get('uploaded_count', 0)
                
                pipeline_tracker.complete_step(self.job_id, "database_upload", success=True)
                
                return {
                    'success': True,
                    'processed_chunks': len(all_chunks),
                    'generated_embeddings': generated_embeddings,
                    'uploaded_vectors': uploaded_vectors,
                    'pipeline_completed_at': datetime.now().isoformat()
                }
            else:
                # No uploaded files, check for URLs
                urls_file = Path("data/raw/urls.txt")
                if urls_file.exists():
                    self.status.logs.append("Found URLs file, starting web scraping...")
                    # This would call the actual scraping pipeline
                    return {'success': False, 'error': 'Web scraping not implemented in simple mode'}
                else:
                    return {'success': False, 'error': 'No data to process - no uploaded files or URLs found'}
                    
        except Exception as e:
            logger.error(f"Simple pipeline failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _run_async_pipeline(self, pipeline, websocket_manager):
        """Run the actual async pipeline with progress tracking"""
        try:
            # Check if we should skip scraping (if using uploaded files only)
            skip_scraping = not Path("data/raw/urls.txt").exists()
            
            # Step 1: Web Scraping
            if not self.should_stop.is_set():
                self.status.current_step = "Web Scraping"
                self.status.progress = 10
                websocket_manager.broadcast(self.status.dict())
                
                if skip_scraping:
                    # Look for existing scraped data or uploaded files
                    from main import find_latest_scraped_data
                    scraped_data_file = find_latest_scraped_data()
                    
                    if not scraped_data_file:
                        # Create a simple data file from uploaded files if any exist
                        upload_dir = Path("data/raw/uploads")
                        if upload_dir.exists() and list(upload_dir.glob("*")):
                            self.status.logs.append("Processing uploaded files...")
                            scraped_data_file = self._create_scraped_data_from_uploads(upload_dir)
                        else:
                            raise Exception("No data to process - no URLs or uploaded files found")
                    
                    scraping_result = {
                        'success': True,
                        'output_file': str(scraped_data_file),
                        'skipped': True,
                        'scraped_pages': 1
                    }
                    self.status.logs.append("Using existing/uploaded data, skipping web scraping")
                else:
                    self.status.logs.append("Starting web scraping...")
                    scraping_result = await pipeline.run_scraping()
                    scraped_data_file = Path(scraping_result.get('output_file', ''))
                
                if not scraping_result.get('success', False):
                    return scraping_result
                
                self.status.progress = 25
                websocket_manager.broadcast(self.status.dict())
            
            # Step 2: Content Processing
            if not self.should_stop.is_set():
                self.status.current_step = "Content Processing"
                self.status.progress = 30
                self.status.logs.append("Starting content processing and chunking...")
                websocket_manager.broadcast(self.status.dict())
                
                processing_result = await pipeline.run_processing(scraped_data_file)
                
                if not processing_result.get('success', False):
                    return processing_result
                
                self.status.progress = 50
                self.status.logs.append(f"Created {processing_result['result'].total_chunks} text chunks")
                websocket_manager.broadcast(self.status.dict())
            
            # Step 3: Embedding Generation
            if not self.should_stop.is_set():
                self.status.current_step = "Embedding Generation"
                self.status.progress = 55
                self.status.logs.append("Generating vector embeddings...")
                websocket_manager.broadcast(self.status.dict())
                
                embedding_result = await pipeline.run_embedding_generation(processing_result)
                
                if not embedding_result.get('success', False):
                    return embedding_result
                
                self.status.progress = 80
                self.status.logs.append(f"Generated {embedding_result['result'].successful_embeddings} embeddings")
                websocket_manager.broadcast(self.status.dict())
            
            # Step 4: Vector Database Upload
            if not self.should_stop.is_set():
                self.status.current_step = "Database Upload"
                self.status.progress = 85
                self.status.logs.append("Uploading vectors to database...")
                websocket_manager.broadcast(self.status.dict())
                
                database_result = await pipeline.run_database_upload(embedding_result)
                
                if not database_result.get('success', False):
                    return database_result
                
                self.status.progress = 95
                self.status.logs.append(f"Uploaded {database_result.get('uploaded_vectors', 0)} vectors")
                websocket_manager.broadcast(self.status.dict())
            
            # Step 5: Validation
            if not self.should_stop.is_set():
                self.status.current_step = "Validation"
                self.status.progress = 98
                self.status.logs.append("Validating pipeline results...")
                websocket_manager.broadcast(self.status.dict())
                
                validation_result = await pipeline.run_validation(database_result)
                
                return {
                    'success': True,
                    'scraping': scraping_result,
                    'processing': processing_result,
                    'embeddings': embedding_result,
                    'database': database_result,
                    'validation': validation_result,
                    'pipeline_completed_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Async pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_scraped_data_from_uploads(self, upload_dir: Path):
        """Create a scraped data file from uploaded files"""
        scraped_data = {
            "scraping_timestamp": datetime.now().isoformat(),
            "source": "uploaded_files",
            "pages": []
        }
        
        # Process uploaded files
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.txt':
                        content = file_path.read_text(encoding='utf-8')
                    elif file_path.suffix.lower() == '.pdf':
                        if HAS_PDF_SUPPORT:
                            content = self._extract_pdf_content(file_path)
                        else:
                            content = f"PDF file: {file_path.name} (PyPDF2 not installed - pip install PyPDF2)"
                    elif file_path.suffix.lower() == '.docx':
                        if HAS_DOCX_SUPPORT:
                            try:
                                content = self._extract_docx_content(file_path)
                            except Exception as docx_error:
                                content = f"DOCX file: {file_path.name} (Error reading file: {str(docx_error)})"
                                logger.error(f"Failed to extract DOCX content from {file_path}: {docx_error}")
                        else:
                            content = f"DOCX file: {file_path.name} (python-docx not installed - pip install python-docx)"
                    else:
                        continue
                    
                    page_data = {
                        "url": f"file://{file_path}",
                        "title": file_path.name,
                        "content": content,
                        "timestamp": datetime.now().isoformat(),
                        "success": True
                    }
                    scraped_data["pages"].append(page_data)
                    
                except Exception as e:
                    logger.error(f"Failed to process uploaded file {file_path}: {e}")
                    continue
        
        # Save the scraped data file
        output_file = Path("data/raw") / f"uploaded_data_{int(time.time())}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save scraped data file: {e}")
            raise
        
        self.status.logs.append(f"Created data file from {len(scraped_data['pages'])} uploaded files")
        return output_file
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text() + "\n"
                
                return content.strip()
        except Exception as e:
            logger.error(f"Failed to extract PDF content from {file_path}: {e}")
            return f"Error extracting PDF content: {str(e)}"
    
    def _extract_docx_content(self, file_path: Path) -> str:
        """Extract text content from DOCX file"""
        try:
            doc = docx.Document(file_path)
            content = ""
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            return content.strip()
        except Exception as e:
            logger.error(f"Failed to extract DOCX content from {file_path}: {e}")
            return f"Error extracting DOCX content: {str(e)}"


# WebSocket manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    def broadcast(self, data: dict):
        """Broadcast data to all connected clients"""
        if not self.active_connections:
            return
            
        message = json.dumps(data)
        for connection in self.active_connections.copy():
            try:
                # Check if there's a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._send_message(connection, message))
                except RuntimeError:
                    # No running event loop, skip WebSocket updates
                    pass
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                self.disconnect(connection)
    
    async def _send_message(self, connection: WebSocket, message: str):
        """Helper method to send message asynchronously"""
        try:
            await connection.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(connection)


# Global instances
websocket_manager = WebSocketManager()
current_job: Optional[PipelineJob] = None


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Tekyz Data Pipeline API", "version": "1.0.0"}


@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status with enhanced metrics"""
    global current_job
    
    try:
        if current_job and current_job.job_id in active_jobs:
            # Return enhanced status for running job
            return current_job._get_enhanced_status()
        
        # Check if any jobs are running in database
        active_jobs_list = pipeline_tracker.get_active_jobs()
        if active_jobs_list:
            latest_job = active_jobs_list[0]
            return {
                "job_id": latest_job.job_id,
                "is_running": True,
                "status": latest_job.status,
                "metrics": latest_job.metrics.__dict__ if latest_job.metrics else None
            }
        
        # No active jobs
        return {
            "is_running": False,
            "progress": 0,
            "current_step": "Idle",
            "logs": [],
            "metrics": None,
            "job_id": None
        }
    
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        return PipelineStatus().dict()


@app.post("/pipeline/start")
async def start_pipeline(
    files: List[UploadFile] = File(default=[]),
    urls: List[str] = Form(default=[])
):
    """Start the data ingestion pipeline"""
    global current_job
    
    # Check if pipeline is already running
    if current_job and current_job.status.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is already running")
    
    # Validate inputs
    if not files and not urls:
        raise HTTPException(status_code=400, detail="No files or URLs provided")
    
    # Read file contents immediately before they get closed
    file_data = []
    for file in files:
        try:
            content = await file.read()
            file_data.append({
                'filename': file.filename,
                'content': content,
                'content_type': file.content_type
            })
        except Exception as e:
            logger.error(f"Failed to read file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read file {file.filename}: {e}")
    
    # Create new job with enhanced tracking
    job_id = str(uuid.uuid4())
    current_job = PipelineJob(job_id)
    active_jobs[job_id] = current_job
    
    # Start the pipeline with file data instead of UploadFile objects
    current_job.start(file_data, urls, websocket_manager)
    
    logger.info(f"Started pipeline job: {job_id}")
    return {
        "message": "Pipeline started with enhanced tracking", 
        "job_id": job_id,
        "input_files": len(file_data),
        "input_urls": len(urls),
        "tracking_enabled": True
    }


@app.post("/pipeline/stop")
async def stop_pipeline():
    """Stop the running pipeline"""
    global current_job
    
    if not current_job or not current_job.status.is_running:
        raise HTTPException(status_code=400, detail="No pipeline is currently running")
    
    current_job.stop()
    logger.info(f"Stopped pipeline job: {current_job.job_id}")
    
    return {"message": "Pipeline stop requested"}


@app.websocket("/ws/pipeline")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# New enhanced tracking endpoints

@app.get("/pipeline/jobs")
async def get_job_history():
    """Get pipeline job history"""
    try:
        jobs = pipeline_tracker.get_job_history(limit=20)
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "start_time": job.start_time,
                    "end_time": job.end_time,
                    "input_urls": job.input_urls,
                    "metrics": job.metrics.__dict__ if job.metrics else None,
                    "error_message": job.error_message
                }
                for job in jobs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job history: {str(e)}")

@app.get("/pipeline/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Get detailed information about a specific job"""
    try:
        job = pipeline_tracker.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "input_urls": job.input_urls,
            "metrics": job.metrics.__dict__ if job.metrics else None,
            "logs": job.logs,
            "error_message": job.error_message,
            "steps": [
                {
                    "name": step.step_name,
                    "order": step.step_order,
                    "status": step.status,
                    "progress": step.progress_percentage,
                    "items_processed": step.items_processed,
                    "items_total": step.items_total,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "duration": step.duration,
                    "error_message": step.error_message
                }
                for step in job.steps
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {str(e)}")

@app.get("/pipeline/stats")
async def get_pipeline_stats():
    """Get overall pipeline statistics"""
    try:
        stats = pipeline_tracker.get_job_stats()
        active_jobs_list = pipeline_tracker.get_active_jobs()
        
        return {
            "statistics": stats,
            "active_jobs": len(active_jobs_list),
            "active_job_ids": [job.job_id for job in active_jobs_list]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline stats: {str(e)}")

@app.get("/pipeline/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
    """Get real-time progress for a specific job"""
    try:
        # Check if job is currently running
        if job_id in active_jobs:
            job = active_jobs[job_id]
            status = job._get_enhanced_status()
            
            # Add step information
            current_step = pipeline_tracker.get_current_step(job_id)
            all_steps = pipeline_tracker.get_job_steps(job_id)
            
            status["current_step_info"] = {
                "name": current_step.step_name if current_step else None,
                "order": current_step.step_order if current_step else None,
                "progress": current_step.progress_percentage if current_step else 0,
                "items_processed": current_step.items_processed if current_step else 0,
                "items_total": current_step.items_total if current_step else 0
            }
            
            status["all_steps"] = [
                {
                    "name": step.step_name,
                    "order": step.step_order,
                    "status": step.status,
                    "progress": step.progress_percentage,
                    "items_processed": step.items_processed,
                    "items_total": step.items_total,
                    "duration": step.duration,
                    "error_message": step.error_message
                }
                for step in all_steps
            ]
            
            return status
        
        # Get from database for completed jobs
        job = pipeline_tracker.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        current_step = pipeline_tracker.get_current_step(job_id)
        all_steps = pipeline_tracker.get_job_steps(job_id)
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "metrics": job.metrics.__dict__ if job.metrics else None,
            "progress": 100 if job.status in ['completed', 'failed'] else 0,
            "logs": job.logs[-10:] if job.logs else [],  # Last 10 logs
            "current_step_info": {
                "name": current_step.step_name if current_step else None,
                "order": current_step.step_order if current_step else None,
                "progress": current_step.progress_percentage if current_step else 0,
                "items_processed": current_step.items_processed if current_step else 0,
                "items_total": current_step.items_total if current_step else 0
            },
            "all_steps": [
                {
                    "name": step.step_name,
                    "order": step.step_order,
                    "status": step.status,
                    "progress": step.progress_percentage,
                    "items_processed": step.items_processed,
                    "items_total": step.items_total,
                    "duration": step.duration,
                    "error_message": step.error_message
                }
                for step in all_steps
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job progress: {str(e)}")

@app.delete("/pipeline/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        if job_id in active_jobs:
            job = active_jobs[job_id]
            job.stop()
            pipeline_tracker.complete_job(job_id, success=False, error_message="Job cancelled by user")
            del active_jobs[job_id]
            return {"message": f"Job {job_id} cancelled"}
        else:
            raise HTTPException(status_code=404, detail="Job not found or not running")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@app.post("/pipeline/cleanup")
async def cleanup_old_jobs(days: int = 30):
    """Clean up old completed jobs"""
    try:
        cleaned_count = pipeline_tracker.cleanup_old_jobs(days)
        return {"message": f"Cleaned up {cleaned_count} jobs older than {days} days"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup jobs: {str(e)}")

@app.get("/pipeline/jobs/{job_id}/visualization")
async def get_job_visualization(job_id: str):
    """Get comprehensive visualization data for a pipeline job"""
    try:
        viz_data = pipeline_tracker.get_pipeline_visualization_data(job_id)
        if not viz_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Convert dataclasses to dicts for JSON serialization
        result = {
            'job': {
                'job_id': viz_data['job'].job_id,
                'status': viz_data['job'].status,
                'start_time': viz_data['job'].start_time,
                'end_time': viz_data['job'].end_time,
                'input_urls': viz_data['job'].input_urls,
                'metrics': viz_data['job'].metrics.__dict__ if viz_data['job'].metrics else None,
                'error_message': viz_data['job'].error_message
            },
            'steps_with_logs': [
                {
                    'step': {
                        'step_name': item['step'].step_name,
                        'step_order': item['step'].step_order,
                        'status': item['step'].status,
                        'start_time': item['step'].start_time,
                        'end_time': item['step'].end_time,
                        'progress_percentage': item['step'].progress_percentage,
                        'items_total': item['step'].items_total,
                        'items_processed': item['step'].items_processed,
                        'duration': item['step'].duration,
                        'error_message': item['step'].error_message
                    },
                    'logs': item['logs']
                }
                for item in viz_data['steps_with_logs']
            ],
            'timeline': viz_data['timeline'],
            'metrics_history': viz_data['metrics_history'],
            'summary': viz_data['summary']
        }
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job visualization: {str(e)}")

@app.get("/pipeline/jobs/{job_id}/step/{step_name}/logs")
async def get_step_logs(job_id: str, step_name: str):
    """Get logs for a specific step"""
    try:
        logs = pipeline_tracker.get_step_logs(job_id, step_name)
        return {"job_id": job_id, "step_name": step_name, "logs": logs}
    except Exception as e:
        logger.error(f"Failed to get step logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get step logs: {str(e)}")

if __name__ == "__main__":
    # Create necessary directories
    directories = ["data/raw", "data/processed", "data/embeddings", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 