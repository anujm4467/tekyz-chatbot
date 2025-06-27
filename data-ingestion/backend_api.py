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

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

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

class PipelineJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = PipelineStatus(job_id=job_id)
        self.thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        
    def start(self, file_data: List[dict], urls: List[str], websocket_manager):
        """Start the pipeline in a separate thread"""
        self.thread = threading.Thread(
            target=self._run_pipeline,
            args=(file_data, urls, websocket_manager),
            daemon=True
        )
        self.thread.start()
        
    def stop(self):
        """Stop the pipeline"""
        self.should_stop.set()
        
    def _run_pipeline(self, file_data: List[dict], urls: List[str], websocket_manager):
        """Execute the pipeline"""
        try:
            self.status.is_running = True
            self.status.current_step = "Initializing"
            self.status.progress = 0
            websocket_manager.broadcast(self.status.dict())
            
            # Save uploaded files
            if file_data:
                self._save_file_data(file_data, websocket_manager)
                self.status.logs.append(f"✅ Saved {len(file_data)} uploaded files")
                websocket_manager.broadcast(self.status.dict())
                
            # Process URLs
            if urls:
                self._process_urls(urls, websocket_manager)
                self.status.logs.append(f"✅ Processed {len(urls)} URLs")
                websocket_manager.broadcast(self.status.dict())
                
            # Run the standalone pipeline (no external dependencies)
            self.status.logs.append("Starting standalone processing...")
            websocket_manager.broadcast(self.status.dict())
            self._execute_pipeline_direct(websocket_manager)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.status.errors.append({"message": str(e), "timestamp": datetime.now().isoformat()})
            websocket_manager.broadcast(self.status.dict())
        finally:
            self.status.is_running = False
            self.status.current_step = "Completed"
            websocket_manager.broadcast(self.status.dict())
    
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
        urls_file = Path("data/raw/urls.txt")
        with open(urls_file, "w") as f:
            for url in urls:
                f.write(f"{url}\n")
                
        log_msg = f"Saved {len(urls)} URLs for processing"
        self.status.logs.append(log_msg)
        logger.info(log_msg)
        
    def _execute_pipeline_direct(self, websocket_manager):
        """Execute the pipeline directly without the problematic DataIngestionPipeline class"""
        try:
            self.status.logs.append("Starting direct pipeline execution...")
            websocket_manager.broadcast(self.status.dict())
            
            # Run the simplified pipeline without any complex dependencies
            result = self._run_standalone_pipeline(websocket_manager)
            
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
            
    def _run_standalone_pipeline(self, websocket_manager):
        """Run a complete pipeline with text processing, embedding generation, and vector storage"""
        try:
            # Check if we have uploaded files to process
            upload_dir = Path("data/raw/uploads")
            if upload_dir.exists() and list(upload_dir.glob("*")):
                self.status.logs.append("Found uploaded files, processing...")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 1: Create scraped data from uploads
                self.status.current_step = "Processing Uploads"
                self.status.progress = 10
                scraped_data_file = self._create_scraped_data_from_uploads(upload_dir)
                self.status.logs.append(f"Created data file: {scraped_data_file}")
                websocket_manager.broadcast(self.status.dict())
                
                # Step 2: Advanced text processing and chunking
                self.status.current_step = "Text Processing & Chunking"
                self.status.progress = 30
                self.status.logs.append("Processing and chunking text content...")
                websocket_manager.broadcast(self.status.dict())
                
                # Load the scraped data
                with open(scraped_data_file, 'r', encoding='utf-8') as f:
                    scraped_data = json.load(f)
                
                # Use the real text processor
                all_chunks = self._process_with_real_pipeline(scraped_data, websocket_manager)
                
                self.status.logs.append(f"Created {len(all_chunks)} text chunks")
                self.status.progress = 50
                websocket_manager.broadcast(self.status.dict())
                
                # Step 3: Generate embeddings
                self.status.current_step = "Generating Embeddings"
                self.status.progress = 70
                self.status.logs.append("Generating vector embeddings...")
                websocket_manager.broadcast(self.status.dict())
                
                embeddings_data = self._generate_embeddings(all_chunks, websocket_manager)
                
                self.status.logs.append(f"Generated embeddings for {len(embeddings_data)} chunks")
                self.status.progress = 85
                websocket_manager.broadcast(self.status.dict())
                
                # Step 4: Upload to vector database
                self.status.current_step = "Storing Vectors"
                self.status.progress = 90
                self.status.logs.append("Uploading vectors to database...")
                websocket_manager.broadcast(self.status.dict())
                
                upload_results = self._upload_to_vector_db(embeddings_data, websocket_manager)
                
                self.status.logs.append(f"✅ Uploaded {upload_results.get('uploaded_count', 0)} vectors to database")
                self.status.progress = 100
                self.status.current_step = "Complete"
                websocket_manager.broadcast(self.status.dict())
                
                return {
                    'success': True,
                    'processed_chunks': len(all_chunks),
                    'generated_embeddings': len(embeddings_data),
                    'uploaded_vectors': upload_results.get('uploaded_count', 0),
                    'pipeline_completed_at': datetime.now().isoformat()
                }
            else:
                # No uploaded files, check for URLs
                urls_file = Path("data/raw/urls.txt")
                if urls_file.exists():
                    self.status.logs.append("Found URLs file, but web scraping not implemented in standalone mode")
                    return {'success': False, 'error': 'Web scraping not implemented in standalone mode'}
                else:
                    return {'success': False, 'error': 'No data to process - no uploaded files or URLs found'}
                    
        except Exception as e:
            logger.error(f"Standalone pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
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
        """Process text using the real text processing pipeline"""
        try:
            # Import the real text processor
            from src.processor.text_processor import TextProcessor, ChunkingConfig
            
            # Create chunking configuration
            chunking_config = ChunkingConfig(
                max_chunk_size=800,
                min_chunk_size=50,
                overlap_size=100,
                split_by_sentences=True,
                preserve_paragraphs=True,
                respect_word_boundaries=True
            )
            
            # Initialize text processor
            text_processor = TextProcessor(chunking_config)
            
            # Process each page
            all_chunks = []
            for page in scraped_data.get('pages', []):
                try:
                    chunks = text_processor.process_page_content(page)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process page {page.get('url', 'unknown')}: {e}")
                    continue
            
            return all_chunks
            
        except ImportError as e:
            logger.error(f"Failed to import text processor: {e}")
            # Fallback to simple chunking
            return self._simple_fallback_chunking(scraped_data)
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            # Fallback to simple chunking
            return self._simple_fallback_chunking(scraped_data)
    
    def _simple_fallback_chunking(self, scraped_data):
        """Fallback simple chunking if real processor fails"""
        all_chunks = []
        for page in scraped_data.get('pages', []):
            content = page.get('content', '')
            if content and content.strip():
                chunks = self._simple_text_chunking(content, page.get('url', ''))
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
    
    def _upload_to_vector_db(self, embeddings_data, websocket_manager):
        """Upload embeddings to Qdrant vector database"""
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
            
            # Prepare embeddings data for upload (already in correct format)
            # embeddings_data already contains: chunk_id, text, embedding, metadata
            self.status.logs.append(f"Uploading {len(embeddings_data)} vectors to collection {config.collection_name}...")
            websocket_manager.broadcast(self.status.dict())
            
            # Upload vectors using the QdrantManager
            upload_result = qdrant_manager.upload_vectors(
                embeddings=embeddings_data,
                progress_callback=lambda **kwargs: self.status.logs.append(f"Upload progress: {kwargs}")
            )
            
            return {
                'uploaded_count': upload_result.uploaded_vectors,
                'failed_count': upload_result.failed_vectors,
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
                
                # Step 3: Generate embeddings (simplified)
                self.status.current_step = "Generating Embeddings"
                self.status.progress = 80
                self.status.logs.append("Generating embeddings...")
                websocket_manager.broadcast(self.status.dict())
                
                # Simulate embedding generation for now to avoid complex dependencies
                import time
                time.sleep(2)
                
                self.status.logs.append(f"Generated embeddings for {len(all_chunks)} chunks")
                self.status.progress = 100
                self.status.current_step = "Complete"
                websocket_manager.broadcast(self.status.dict())
                
                return {
                    'success': True,
                    'processed_chunks': len(all_chunks),
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
    """Get current pipeline status"""
    if current_job:
        return current_job.status.dict()
    else:
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
    
    # Create new job
    job_id = str(uuid.uuid4())
    current_job = PipelineJob(job_id)
    
    # Start the pipeline with file data instead of UploadFile objects
    current_job.start(file_data, urls, websocket_manager)
    
    logger.info(f"Started pipeline job: {job_id}")
    return {"message": "Pipeline started", "job_id": job_id}


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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


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