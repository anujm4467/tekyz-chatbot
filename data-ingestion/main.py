#!/usr/bin/env python3
"""
Main entry point for the Tekyz Data Ingestion Pipeline

This script orchestrates the complete data ingestion process:
1. Web scraping from tekyz.com (with skip mechanism if data exists)
2. Content processing and chunking
3. Embedding generation
4. Vector database storage
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any, List
import time
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import get_settings
from scraper.enhanced_orchestrator import EnhancedScrapingOrchestrator
from processor.pipeline import ProcessingPipeline, ProcessingConfig
from processor.text_processor import ChunkingConfig
from embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from database.qdrant_client import QdrantManager
from database.vector_db_manager import VectorDBManager
from database.models import VectorConfig


def setup_logging():
    """Set up logging configuration"""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Add file handler
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days"
    )
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    )


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "data/embeddings",
        "data/embeddings/cache",
        "models/sentence_transformers",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def find_latest_scraped_data() -> Optional[Path]:
    """Find the most recent scraped data file"""
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return None
    
    # Look for scraped data files
    scraped_files = list(data_dir.glob("tekyz_scraped_data_*.json"))
    if not scraped_files:
        return None
    
    # Return the most recent file
    latest_file = max(scraped_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Found existing scraped data: {latest_file}")
    return latest_file


class DataIngestionPipeline:
    """Main data ingestion pipeline orchestrator"""
    
    def __init__(self):
        self.settings = get_settings()
        self.scraping_orchestrator = EnhancedScrapingOrchestrator()
        
        # Initialize processing components
        self.processing_config = ProcessingConfig(
            chunking=ChunkingConfig(
                max_chunk_size=1024,
                min_chunk_size=100,
                overlap_size=50,
                split_by_sentences=True,
                preserve_paragraphs=True,
                respect_word_boundaries=True
            ),
            min_quality_score=0.6,
            similarity_threshold=0.85,
            max_workers=4,
            batch_size=50,
            output_dir=Path("data/processed")
        )
        
        self.embedding_config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            max_seq_length=512,
            normalize_embeddings=True,
            device="auto",
            cache_embeddings=True,
            cache_dir=Path("data/embeddings/cache"),
            model_cache_dir=Path("models/sentence_transformers")
        )
        
        logger.info("Data Ingestion Pipeline initialized")
    
    async def run_full_pipeline(self, skip_scraping: bool = False, force_scraping: bool = False):
        """Execute the complete data ingestion pipeline"""
        logger.info("Starting full data ingestion pipeline")
        
        try:
            # Step 1: Check if tekyz data already exists (unless forced)
            if not force_scraping:
                logger.info("Checking if tekyz.com data already exists...")
                data_check = await self.check_existing_tekyz_data()
                
                if data_check.get('exists', False):
                    logger.info("✅ Tekyz.com data already exists in database!")
                    logger.info(f"Found {data_check.get('vector_count', 0)} vectors, {data_check.get('tekyz_pages', 0)} tekyz pages")
                    
                    return {
                        'success': True,
                        'skipped_reason': 'Data already exists',
                        'existing_data': data_check,
                        'pipeline_completed_at': datetime.now().isoformat()
                    }
            
            # Step 2: Web Scraping (with skip mechanism)
            scraped_data_file = None
            if skip_scraping and not force_scraping:
                scraped_data_file = find_latest_scraped_data()
                if scraped_data_file:
                    logger.info(f"Skipping scraping, using existing data: {scraped_data_file}")
                    scraping_result = {
                        'success': True,
                        'output_file': str(scraped_data_file),
                        'skipped': True
                    }
                else:
                    logger.warning("No existing scraped data found, proceeding with scraping")
                    skip_scraping = False
            
            if not skip_scraping:
                logger.info("Step 1: Starting web scraping")
                scraping_result = await self.run_scraping()
                
                if not scraping_result.get('success', False):
                    logger.error("Web scraping failed, stopping pipeline")
                    return scraping_result
                
                scraped_data_file = Path(scraping_result.get('output_file', ''))
            
            # Step 2: Content Processing
            logger.info("Step 2: Starting content processing")
            processing_result = await self.run_processing(scraped_data_file)
            
            if not processing_result.get('success', False):
                logger.error("Content processing failed, stopping pipeline")
                return processing_result
            
            # Step 3: Embedding Generation
            logger.info("Step 3: Starting embedding generation")
            embedding_result = await self.run_embedding_generation(processing_result)
            
            if not embedding_result.get('success', False):
                logger.error("Embedding generation failed, stopping pipeline")
                return embedding_result
            
            # Step 4: Vector Database Upload
            logger.info("Step 4: Starting vector database upload")
            database_result = await self.run_database_upload(embedding_result)
            
            if not database_result.get('success', False):
                logger.error("Database upload failed, stopping pipeline")
                return database_result
            
            # Step 5: Validation
            logger.info("Step 5: Starting validation")
            validation_result = await self.run_validation(database_result)
            
            logger.success("Data ingestion pipeline completed successfully!")
            
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
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    async def run_scraping(self):
        """Run the web scraping component"""
        logger.info("Executing web scraping component")
        
        try:
            # Run the enhanced scraping orchestrator
            result = await self.scraping_orchestrator.run_comprehensive_scrape(
                max_discovery_depth=5,
                max_discovery_pages=500,
                max_scraping_concurrent=5,
                enable_content_extraction=True
            )
            
            if result.get('success', False):
                logger.success(f"Scraping completed successfully! "
                             f"Scraped {result.get('scraped_pages', 0)} pages")
                
                # Log some statistics
                stats = result.get('stats', {})
                logger.info(f"Scraping statistics:")
                logger.info(f"  - Success rate: {stats.get('success_rate', 0):.1f}%")
                logger.info(f"  - Duration: {stats.get('duration_seconds', 0):.2f} seconds")
                logger.info(f"  - Total content: {stats.get('total_content_length', 0)} characters")
                
            else:
                logger.error(f"Scraping failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Scraping component failed: {str(e)}")
            raise
    
    async def run_processing(self, scraped_data_file: Path):
        """Run the content processing component"""
        logger.info(f"Processing scraped data from: {scraped_data_file}")
        
        try:
            # Initialize processing pipeline
            processing_pipeline = ProcessingPipeline(self.processing_config)
            
            # Process the scraped data
            def progress_callback(stage: str, current: int, total: int):
                logger.info(f"Processing {stage}: {current}/{total} ({current/total*100:.1f}%)")
            
            result = processing_pipeline.process_scraped_data(
                scraped_data_file, 
                progress_callback=progress_callback
            )
            
            if result.total_chunks > 0:
                logger.success(f"Content processing completed successfully!")
                logger.info(f"Processing statistics:")
                logger.info(f"  - Total pages: {result.total_pages}")
                logger.info(f"  - Total chunks: {result.total_chunks}")
                logger.info(f"  - Valid chunks: {result.valid_chunks}")
                logger.info(f"  - Filtered chunks: {result.filtered_chunks}")
                logger.info(f"  - Processing time: {result.processing_time_seconds:.2f} seconds")
                
                return {
                    'success': True,
                    'result': result,
                    'output_files': result.output_files
                }
            else:
                logger.error("No chunks were created during processing")
                return {
                    'success': False,
                    'error': 'No chunks created',
                    'result': result
                }
                
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def run_embedding_generation(self, processing_result: Dict[str, Any]):
        """Run the embedding generation component"""
        logger.info("Generating embeddings for processed chunks")
        
        try:
            # Initialize embedding generator
            embedding_generator = EmbeddingGenerator(self.embedding_config)
            
            # Get processed chunks from the processing result
            processing_data = processing_result.get('result')
            if not processing_data or not processing_data.output_files:
                logger.error("No processed data files found")
                return {'success': False, 'error': 'No processed data files'}
            
            # Load processed chunks from the output files
            chunks = []
            for output_file in processing_data.output_files:
                if 'chunks' in output_file:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        file_chunks = json.load(f)
                        chunks.extend(file_chunks)
            
            if not chunks:
                logger.error("No chunks found in processed data files")
                return {'success': False, 'error': 'No chunks found'}
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Progress callback
            def progress_callback(current: int, total: int):
                logger.info(f"Embedding generation: {current}/{total} ({current/total*100:.1f}%)")
            
            # Generate embeddings
            embedding_result = embedding_generator.generate_embeddings(
                chunks, 
                progress_callback=progress_callback
            )
            
            if embedding_result.successful_embeddings > 0:
                logger.success(f"Embedding generation completed successfully!")
                logger.info(f"Embedding statistics:")
                logger.info(f"  - Total chunks: {embedding_result.total_chunks}")
                logger.info(f"  - Successful embeddings: {embedding_result.successful_embeddings}")
                logger.info(f"  - Failed embeddings: {embedding_result.failed_embeddings}")
                logger.info(f"  - Processing time: {embedding_result.total_processing_time_seconds:.2f} seconds")
                logger.info(f"  - Average time per chunk: {embedding_result.average_time_per_chunk_ms:.2f} ms")
                
                # Save embeddings to file
                embeddings_file = Path("data/embeddings") / f"tekyz_embeddings_{int(time.time())}.json"
                embedding_generator.save_embeddings(
                    embedding_result.embeddings,
                    embeddings_file,
                    include_text=True
                )
                
                return {
                    'success': True,
                    'result': embedding_result,
                    'embeddings_file': str(embeddings_file)
                }
            else:
                logger.error("No embeddings were generated successfully")
                return {
                    'success': False,
                    'error': 'No successful embeddings',
                    'result': embedding_result
                }
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def run_database_upload(self, embedding_result: Dict[str, Any]):
        """Run the database upload component with merge into tekyz_knowledge"""
        logger.info("Uploading embeddings to vector database")
        
        try:
            # Initialize vector database components
            from database.models import VectorConfig
            
            vector_config = VectorConfig(
                collection_name="tekyz_knowledge",
                host="localhost",
                port=6333
            )
            
            vector_db = QdrantManager(config=vector_config)
            
            # Check database health
            if not vector_db.connect():
                logger.error("Vector database is not available")
                return {
                    'success': False,
                    'error': 'Vector database not available'
                }
            
            # Extract embeddings from the result - handle different result structures
            embeddings = None
            
            # Try different possible structures for embedding results
            if 'result' in embedding_result and hasattr(embedding_result['result'], 'embeddings'):
                embeddings = embedding_result['result'].embeddings
                logger.info("Found embeddings in result.embeddings")
            elif 'result' in embedding_result and 'embeddings' in embedding_result['result']:
                embeddings = embedding_result['result']['embeddings']
                logger.info("Found embeddings in result['embeddings']")
            elif 'embeddings' in embedding_result:
                embeddings = embedding_result['embeddings']
                logger.info("Found embeddings in direct embeddings key")
            elif isinstance(embedding_result, dict) and 'result' in embedding_result:
                # Handle case where result is a BatchEmbeddingResult object
                result_obj = embedding_result['result']
                if hasattr(result_obj, 'embeddings'):
                    embeddings = result_obj.embeddings
                    logger.info("Found embeddings in result object")
            
            if not embeddings:
                logger.error("No embeddings found in result")
                logger.error(f"Available keys in embedding_result: {list(embedding_result.keys())}")
                if 'result' in embedding_result:
                    result_obj = embedding_result['result']
                    if hasattr(result_obj, '__dict__'):
                        logger.error(f"Available attributes in result: {list(result_obj.__dict__.keys())}")
                    elif isinstance(result_obj, dict):
                        logger.error(f"Available keys in result dict: {list(result_obj.keys())}")
                
                return {
                    'success': False,
                    'error': 'No embeddings provided'
                }
            
            # Always use tekyz_knowledge collection for merge
            collection_name = "tekyz_knowledge"
            embedding_dim = len(embeddings[0].embedding) if embeddings and hasattr(embeddings[0], 'embedding') else 384
            
            # Fix embedding dimension detection
            if embeddings:
                try:
                    first_embedding = embeddings[0]
                    if hasattr(first_embedding, 'embedding'):
                        if isinstance(first_embedding.embedding, list):
                            embedding_dim = len(first_embedding.embedding)
                        elif hasattr(first_embedding.embedding, 'shape'):
                            embedding_dim = first_embedding.embedding.shape[0]
                        else:
                            embedding_dim = len(first_embedding.embedding)
                    elif hasattr(first_embedding, 'vector'):
                        embedding_dim = len(first_embedding.vector)
                    logger.info(f"Detected embedding dimension: {embedding_dim}")
                except Exception as e:
                    logger.warning(f"Could not detect embedding dimension: {str(e)}, using default 384")
                    embedding_dim = 384
            
            logger.info(f"📤 Uploading {len(embeddings)} embeddings to collection '{collection_name}'")
            logger.info(f"🔄 Using MERGE mode - data will be added to existing collection")
            
            # Create vector configuration for tekyz knowledge base
            from database.models import DistanceMetric
            vector_config = VectorConfig(
                collection_name=collection_name,
                vector_size=embedding_dim,
                distance_metric=DistanceMetric.COSINE
            )
            
            # Use vector database manager for intelligent merging
            from database.vector_db_manager import VectorDBManager, IngestionConfig
            vector_db_manager = VectorDBManager(vector_db=vector_db)
            
            # Configure ingestion for merge mode
            ingestion_config = IngestionConfig(
                collection_name=collection_name,
                vector_size=embedding_dim,
                distance_metric="Cosine",
                batch_size=100,
                max_workers=4,
                enable_compression=True,
                backup_enabled=True,
                quality_threshold=0.5
            )
            
            # Check existing data
            existing_data = vector_db_manager.check_tekyz_data_exists(collection_name)
            if existing_data.get('exists', False):
                logger.info(f"📊 Found existing data: {existing_data.get('vector_count', 0)} vectors, "
                           f"{existing_data.get('tekyz_pages', 0)} tekyz pages")
            else:
                logger.info("📊 No existing data found - this will be the first ingestion")
            
            # Enhance embeddings with consistent Tekyz metadata
            enhanced_embeddings = []
            for i, embedding in enumerate(embeddings):
                try:
                    # Create a copy of the embedding with enhanced metadata
                    if hasattr(embedding, 'to_dict'):
                        embedding_dict = embedding.to_dict()
                    elif hasattr(embedding, '__dict__'):
                        embedding_dict = embedding.__dict__.copy()
                    else:
                        # Handle case where embedding is already a dict
                        embedding_dict = dict(embedding) if isinstance(embedding, dict) else {}
                    
                    # Ensure proper metadata structure for Tekyz content
                    if 'metadata' not in embedding_dict:
                        embedding_dict['metadata'] = {}
                    
                    metadata = embedding_dict['metadata']
                    
                    # Add tekyz.com identifier to metadata for better detection
                    if 'url' in metadata and 'tekyz.com' not in metadata.get('url', ''):
                        # If URL doesn't contain tekyz.com, add it to source_domain
                        metadata['source_domain'] = 'tekyz.com'
                    
                    # Ensure we have source identification
                    if 'source' not in metadata:
                        metadata['source'] = 'tekyz.com'
                    
                    # Add tekyz identifier for easy filtering
                    metadata['company'] = 'tekyz'
                    metadata['content_type'] = metadata.get('content_type', 'website')
                    
                    # Ensure we have proper URL field structure
                    if 'page_url' not in metadata and 'url' in metadata:
                        metadata['page_url'] = metadata['url']
                    elif 'url' not in metadata and 'page_url' in metadata:
                        metadata['url'] = metadata['page_url']
                    
                    enhanced_embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Could not enhance embedding {i}: {str(e)}, using original")
                    enhanced_embeddings.append(embedding)
            
            # Ingest embeddings with merge mode
            def progress_callback(current: int, total: int):
                logger.info(f"Upload progress: {current}/{total} batches ({current/total*100:.1f}%)")
            
            ingestion_result = vector_db_manager.ingest_embeddings(
                records=enhanced_embeddings,
                config=ingestion_config,
                progress_callback=progress_callback,
                merge_mode=True  # Always merge into existing collection
            )
            
            if ingestion_result.successful_insertions > 0:
                logger.success(f"✅ Database upload completed successfully!")
                logger.info(f"📊 Upload statistics:")
                logger.info(f"  - New vectors added: {ingestion_result.successful_insertions}")
                logger.info(f"  - Failed uploads: {ingestion_result.failed_insertions}")
                logger.info(f"  - Processing time: {ingestion_result.processing_time_seconds:.2f}s")
                logger.info(f"  - Collection: {collection_name}")
                logger.info(f"  - Vector dimension: {embedding_dim}")
                
                if ingestion_result.collection_stats:
                    logger.info(f"  - Total vectors in collection: {ingestion_result.collection_stats.points_count}")
                
                return {
                    'success': True,
                    'uploaded_vectors': ingestion_result.successful_insertions,
                    'failed_vectors': ingestion_result.failed_insertions,
                    'total_vectors_in_collection': ingestion_result.collection_stats.points_count if ingestion_result.collection_stats else 0,
                    'processing_time': ingestion_result.processing_time_seconds,
                    'collection_name': collection_name,
                    'vector_dimension': embedding_dim,
                    'merge_mode': True
                }
            else:
                logger.error("❌ No vectors were uploaded successfully")
                return {
                    'success': False,
                    'error': 'No vectors uploaded',
                    'ingestion_result': ingestion_result.to_dict()
                }
                
        except Exception as e:
            logger.error(f"Database upload failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def run_validation(self, database_result: Dict[str, Any]):
        """Run the validation component"""
        logger.info("Validating pipeline results")
        
        try:
            # Initialize vector database for validation
            from database.models import VectorConfig
            vector_config = VectorConfig(
                collection_name="tekyz_knowledge",
                host="localhost",
                port=6333
            )
            vector_db = QdrantManager(config=vector_config)
            
            # Check database connection
            if not vector_db.connect():
                logger.error("Vector database is not available for validation")
                return {'success': False, 'error': 'Database not available'}
            
            collection_name = database_result.get('collection_name', 'tekyz_knowledge')
            
            # Get collection info
            collection_info = vector_db.get_collection_info()
            
            if collection_info:
                vector_count = collection_info.points_count
                logger.info(f"Validation results:")
                logger.info(f"  - Collection: {collection_name}")
                logger.info(f"  - Vectors in database: {vector_count}")
                logger.info(f"  - Expected vectors: {database_result.get('uploaded_vectors', 0)}")
                
                # Test a simple search with a dummy vector
                import numpy as np
                test_vector = np.random.rand(database_result.get('vector_dimension', 384)).tolist()
                search_results = vector_db.search_vectors(
                    query_vector=test_vector,
                    limit=3
                )
                
                if search_results:
                    logger.info(f"  - Test search successful: found {len(search_results)} results")
                    logger.success("Pipeline validation completed successfully!")
                    
                    return {
                        'success': True,
                        'collection_info': {
                            'name': getattr(collection_info, 'name', collection_name),
                            'vectors_count': getattr(collection_info, 'vectors_count', 0),
                            'points_count': getattr(collection_info, 'points_count', 0)
                        },
                        'test_search_results': len(search_results),
                        'validation_passed': True
                    }
                else:
                    logger.warning("Test search returned no results")
                    return {
                        'success': True,
                        'collection_info': {
                            'name': getattr(collection_info, 'name', collection_name),
                            'vectors_count': getattr(collection_info, 'vectors_count', 0),
                            'points_count': getattr(collection_info, 'points_count', 0)
                        },
                        'test_search_results': 0,
                        'validation_passed': False,
                        'warning': 'Test search returned no results'
                    }
            else:
                logger.error(f"Collection {collection_name} not found")
                return {
                    'success': False,
                    'error': f'Collection {collection_name} not found'
                }
                
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def check_existing_tekyz_data(self):
        """Check if tekyz.com data already exists in the database"""
        try:
            from database.models import VectorConfig
            
            # Create proper config for QdrantManager
            vector_config = VectorConfig(
                collection_name="tekyz_knowledge",
                host="localhost", 
                port=6333
            )
            
            # Initialize vector database components
            vector_db = QdrantManager(config=vector_config)
            
            # Check if database is available
            if not vector_db.connect():
                logger.warning("Vector database not available, proceeding with scraping")
                return {'exists': False, 'reason': 'Database not available'}
            
            vector_db_manager = VectorDBManager(vector_db=vector_db)
            
            # Check for existing tekyz data
            data_status = vector_db_manager.check_tekyz_data_exists()
            
            return data_status
            
        except Exception as e:
            logger.warning(f"Failed to check existing data: {str(e)}, proceeding with scraping")
            return {'exists': False, 'reason': f'Error checking: {str(e)}'}


async def main():
    """Main entry point"""
    setup_logging()
    create_directories()
    
    logger.info("Starting Tekyz Data Ingestion Pipeline")
    
    # Check if we should skip scraping
    skip_scraping = len(sys.argv) > 1 and sys.argv[1] == "--skip-scraping"
    
    if skip_scraping:
        logger.info("Skip scraping mode enabled - will use existing data if available")
    
    pipeline = DataIngestionPipeline()
    result = await pipeline.run_full_pipeline(skip_scraping=skip_scraping)
    
    if result.get('success', False):
        logger.success("🎉 Pipeline completed successfully!")
        
        # Print summary
        logger.info("📊 Pipeline Summary:")
        if 'scraping' in result:
            scraping = result['scraping']
            if scraping.get('skipped'):
                logger.info("  ✅ Scraping: Skipped (used existing data)")
            else:
                logger.info(f"  ✅ Scraping: {scraping.get('scraped_pages', 0)} pages")
        
        if 'processing' in result:
            processing = result['processing']['result']
            logger.info(f"  ✅ Processing: {processing.total_chunks} chunks created")
        
        if 'embeddings' in result:
            embeddings = result['embeddings']['result']
            logger.info(f"  ✅ Embeddings: {embeddings.successful_embeddings} embeddings generated")
        
        if 'database' in result:
            database = result['database']
            logger.info(f"  ✅ Database: {database.get('uploaded_vectors', 0)} vectors uploaded")
        
        if 'validation' in result:
            validation = result['validation']
            if validation.get('validation_passed'):
                logger.info("  ✅ Validation: All checks passed")
            else:
                logger.info("  ⚠️  Validation: Some checks failed")
        
    else:
        logger.error("❌ Pipeline failed!")
        if 'error' in result:
            logger.error(f"Error: {result['error']}")
    
    return result


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result.get('success', False) else 1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {str(e)}")
        sys.exit(1) 