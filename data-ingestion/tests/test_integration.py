"""
Integration Tests for Complete Data Ingestion Pipeline

Tests the full end-to-end pipeline from web scraping to vector database storage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import time
from typing import List, Dict, Any

from src.scraper import ScrapingConfig, ScrapingResult
from src.processor import ProcessingConfig, ProcessingStats
from src.embeddings import EmbeddingRecord
from src.database import (
    QdrantVectorDB, VectorDBManager, IntegrationPipeline,
    PipelineConfig, PipelineResult, IngestionConfig
)


class TestIntegrationPipeline:
    """Test the complete integration pipeline."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock vector database for testing."""
        mock_db = Mock(spec=QdrantVectorDB)
        mock_db.collection_exists.return_value = False
        mock_db.create_collection.return_value = True
        mock_db.insert_vectors.return_value = {
            'inserted_count': 10,
            'collection_name': 'test_collection',
            'operation_id': 'test_op_123',
            'status': 'completed'
        }
        mock_db.get_health.return_value = {
            'status': 'healthy',
            'host': 'localhost',
            'port': 6333
        }
        return mock_db
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration."""
        return PipelineConfig(
            scraping=ScrapingConfig(
                base_url="https://example.com",
                max_pages=5,
                delay_range=(0.1, 0.2),
                max_retries=2
            ),
            processing=ProcessingConfig(
                min_chunk_size=100,
                max_chunk_size=500,
                chunk_overlap=50,
                quality_threshold=0.7
            ),
            embedding_model="all-MiniLM-L6-v2",
            embedding_batch_size=2,
            collection_name="test_collection",
            max_concurrent_workers=2,
            save_intermediate_results=True,
            validate_embeddings=True,
            backup_data=True
        )
    
    @pytest.fixture
    def sample_urls(self):
        """Sample URLs for testing."""
        return [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]
    
    @pytest.fixture
    def sample_scraped_content(self):
        """Sample scraped content for testing."""
        return {
            'extracted_content': [
                {
                    'url': 'https://example.com/page1',
                    'title': 'Test Page 1',
                    'content': 'This is test content for page 1. ' * 20,
                    'metadata': {'source': 'web', 'type': 'page'}
                },
                {
                    'url': 'https://example.com/page2', 
                    'title': 'Test Page 2',
                    'content': 'This is test content for page 2. ' * 25,
                    'metadata': {'source': 'web', 'type': 'page'}
                }
            ],
            'successful_scrapes': 2,
            'failed_scrapes': 1,
            'total_time': 1.5
        }
    
    @pytest.fixture
    def sample_processed_chunks(self):
        """Sample processed chunks for testing."""
        return {
            'processed_chunks': [
                {
                    'id': 'chunk_1',
                    'text': 'This is the first chunk of processed content. ' * 10,
                    'metadata': {
                        'url': 'https://example.com/page1',
                        'chunk_index': 0,
                        'quality_score': 0.85
                    }
                },
                {
                    'id': 'chunk_2', 
                    'text': 'This is the second chunk of processed content. ' * 12,
                    'metadata': {
                        'url': 'https://example.com/page1',
                        'chunk_index': 1,
                        'quality_score': 0.90
                    }
                },
                {
                    'id': 'chunk_3',
                    'text': 'This is the third chunk from another page. ' * 8,
                    'metadata': {
                        'url': 'https://example.com/page2',
                        'chunk_index': 0,
                        'quality_score': 0.75
                    }
                }
            ],
            'total_chunks': 3,
            'source_items': 2
        }
    
    @pytest.fixture
    def sample_embedding_records(self):
        """Sample embedding records for testing."""
        return [
            EmbeddingRecord(
                id='chunk_1',
                text='This is the first chunk of processed content. ' * 10,
                embedding=np.random.rand(384),
                metadata={'url': 'https://example.com/page1', 'chunk_index': 0},
                timestamp='2024-01-01T12:00:00',
                source='pipeline',
                embedding_model='all-MiniLM-L6-v2',
                embedding_dimensions=384
            ),
            EmbeddingRecord(
                id='chunk_2',
                text='This is the second chunk of processed content. ' * 12,
                embedding=np.random.rand(384),
                metadata={'url': 'https://example.com/page1', 'chunk_index': 1},
                timestamp='2024-01-01T12:00:01',
                source='pipeline',
                embedding_model='all-MiniLM-L6-v2',
                embedding_dimensions=384
            )
        ]
    
    def test_pipeline_initialization(self, pipeline_config, temp_storage_dir):
        """Test pipeline initialization with all components."""
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant:
            mock_qdrant.return_value.get_collections.return_value = []
            
            pipeline = IntegrationPipeline(
                config=pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            assert pipeline.config == pipeline_config
            assert pipeline.storage_dir == temp_storage_dir
            assert pipeline.pipeline_state['current_stage'] is None
            assert 'web_scraper' in dir(pipeline)
            assert 'content_processor' in dir(pipeline)
            assert 'embedding_generator' in dir(pipeline)
            assert 'vector_db_manager' in dir(pipeline)
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_scraping_stage(self, mock_qdrant, pipeline_config, temp_storage_dir, sample_urls):
        """Test the scraping stage of the pipeline."""
        mock_qdrant.return_value.get_collections.return_value = []
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Mock the web scraper
        with patch.object(pipeline.web_scraper, 'scrape_urls') as mock_scrape:
            mock_scrape.return_value = {
                'extracted_content': [
                    {'url': url, 'title': f'Title {i}', 'content': f'Content {i}'}
                    for i, url in enumerate(sample_urls)
                ],
                'successful_scrapes': len(sample_urls),
                'failed_scrapes': 0
            }
            
            result = pipeline._run_scraping_stage(sample_urls, None)
            
            assert result['successful_scrapes'] == len(sample_urls)
            assert len(result['extracted_content']) == len(sample_urls)
            mock_scrape.assert_called_once_with(sample_urls)
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_processing_stage(self, mock_qdrant, pipeline_config, temp_storage_dir, sample_scraped_content):
        """Test the content processing stage."""
        mock_qdrant.return_value.get_collections.return_value = []
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Mock the content processor
        with patch.object(pipeline.content_processor, 'process_content') as mock_process:
            mock_process.side_effect = [
                [{'id': 'chunk_1', 'text': 'Chunk 1', 'metadata': {}}],
                [{'id': 'chunk_2', 'text': 'Chunk 2', 'metadata': {}}]
            ]
            
            result = pipeline._run_processing_stage(
                sample_scraped_content['extracted_content'],
                None
            )
            
            assert result['total_chunks'] == 2
            assert len(result['processed_chunks']) == 2
            assert mock_process.call_count == 2
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_embedding_stage(self, mock_qdrant, pipeline_config, temp_storage_dir, sample_processed_chunks):
        """Test the embedding generation stage."""
        mock_qdrant.return_value.get_collections.return_value = []
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Mock the embedding components
        with patch.object(pipeline.model_manager, 'load_model') as mock_load_model, \
             patch.object(pipeline.text_preprocessor, 'preprocess_batch') as mock_preprocess, \
             patch.object(pipeline.embedding_generator, 'generate_embeddings') as mock_generate, \
             patch.object(pipeline.embedding_validator, 'validate_embeddings') as mock_validate, \
             patch.object(pipeline.embedding_storage, 'store_embeddings') as mock_store:
            
            # Setup mocks
            mock_model = Mock()
            mock_load_model.return_value = mock_model
            mock_preprocess.return_value = ['text1', 'text2', 'text3']
            mock_generate.return_value = {
                'embeddings': [np.random.rand(384) for _ in range(3)],
                'generation_time': 1.0,
                'statistics': {'total_tokens': 100}
            }
            mock_validate.return_value = {'overall_valid': True, 'issues': []}
            mock_store.return_value = {'stored_records': 3, 'file_paths': ['test.pkl']}
            
            result = pipeline._run_embedding_stage(
                sample_processed_chunks['processed_chunks'],
                None
            )
            
            assert result['total_embeddings'] == 3
            assert len(result['embedding_records']) == 3
            assert result['validation_results']['overall_valid'] is True
            
            # Verify all components were called
            mock_load_model.assert_called_once()
            mock_preprocess.assert_called_once()
            mock_generate.assert_called_once()
            mock_validate.assert_called_once()
            mock_store.assert_called_once()
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_ingestion_stage(self, mock_qdrant, pipeline_config, temp_storage_dir, 
                            sample_embedding_records, mock_vector_db):
        """Test the vector database ingestion stage."""
        mock_qdrant.return_value = mock_vector_db
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Mock the vector DB manager
        with patch.object(pipeline.vector_db_manager, 'ingest_embeddings') as mock_ingest:
            from src.database.vector_db_manager import IngestionResult
            
            mock_ingest.return_value = IngestionResult(
                total_processed=2,
                successful_insertions=2,
                failed_insertions=0,
                processing_time_seconds=0.5,
                errors=[]
            )
            
            result = pipeline._run_ingestion_stage(sample_embedding_records, None)
            
            assert result.successful_insertions == 2
            assert result.failed_insertions == 0
            assert len(result.errors) == 0
            
            mock_ingest.assert_called_once()
            
            # Verify ingestion config
            call_args = mock_ingest.call_args
            ingestion_config = call_args[1]['config']
            assert ingestion_config.collection_name == pipeline_config.collection_name
            assert ingestion_config.vector_size == 384
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_complete_pipeline_success(self, mock_qdrant, pipeline_config, temp_storage_dir, 
                                     sample_urls, mock_vector_db):
        """Test successful complete pipeline execution."""
        mock_qdrant.return_value = mock_vector_db
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Mock all pipeline stages
        with patch.object(pipeline, '_run_scraping_stage') as mock_scraping, \
             patch.object(pipeline, '_run_processing_stage') as mock_processing, \
             patch.object(pipeline, '_run_embedding_stage') as mock_embedding, \
             patch.object(pipeline, '_run_ingestion_stage') as mock_ingestion:
            
            # Setup stage results
            mock_scraping.return_value = {
                'extracted_content': [{'url': url, 'content': f'Content {i}'} for i, url in enumerate(sample_urls)]
            }
            
            mock_processing.return_value = {
                'processed_chunks': [{'id': f'chunk_{i}', 'text': f'Chunk {i}'} for i in range(5)]
            }
            
            mock_embedding.return_value = {
                'embedding_records': [Mock() for _ in range(5)]
            }
            
            from src.database.vector_db_manager import IngestionResult
            mock_ingestion.return_value = IngestionResult(
                total_processed=5,
                successful_insertions=5,
                failed_insertions=0,
                processing_time_seconds=1.0,
                errors=[]
            )
            
            # Run complete pipeline
            result = pipeline.run_complete_pipeline(sample_urls)
            
            # Verify results
            assert result.total_urls_processed == len(sample_urls)
            assert result.total_content_extracted == len(sample_urls)
            assert result.total_chunks_created == 5
            assert result.total_embeddings_generated == 5
            assert result.total_vectors_stored == 5
            assert len(result.errors) == 0
            
            # Verify all stages were called
            mock_scraping.assert_called_once()
            mock_processing.assert_called_once()
            mock_embedding.assert_called_once()
            mock_ingestion.assert_called_once()
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_pipeline_with_progress_callback(self, mock_qdrant, pipeline_config, temp_storage_dir, 
                                           sample_urls, mock_vector_db):
        """Test pipeline with progress callback functionality."""
        mock_qdrant.return_value = mock_vector_db
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Mock progress callback
        progress_callback = Mock()
        
        # Mock pipeline stages to succeed quickly
        with patch.object(pipeline, '_run_scraping_stage') as mock_scraping, \
             patch.object(pipeline, '_run_processing_stage') as mock_processing, \
             patch.object(pipeline, '_run_embedding_stage') as mock_embedding, \
             patch.object(pipeline, '_run_ingestion_stage') as mock_ingestion:
            
            # Setup minimal successful results
            mock_scraping.return_value = {'extracted_content': [{'url': 'test', 'content': 'test'}]}
            mock_processing.return_value = {'processed_chunks': [{'id': 'test', 'text': 'test'}]}
            mock_embedding.return_value = {'embedding_records': [Mock()]}
            
            from src.database.vector_db_manager import IngestionResult
            mock_ingestion.return_value = IngestionResult(
                total_processed=1, successful_insertions=1, failed_insertions=0,
                processing_time_seconds=0.1, errors=[]
            )
            
            # Run pipeline with progress callback
            result = pipeline.run_complete_pipeline(sample_urls, progress_callback)
            
            # Verify progress callback was called for each stage
            assert progress_callback.call_count >= 4  # At least once per stage
            
            # Verify callback was called with stage names
            stage_calls = [call[0][0] for call in progress_callback.call_args_list]
            expected_stages = ['scraping', 'processing', 'embedding', 'ingestion']
            for stage in expected_stages:
                assert stage in stage_calls
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_pipeline_error_handling(self, mock_qdrant, pipeline_config, temp_storage_dir, 
                                   sample_urls, mock_vector_db):
        """Test pipeline error handling and recovery."""
        mock_qdrant.return_value = mock_vector_db
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Mock scraping stage to fail
        with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
            mock_scraping.side_effect = Exception("Scraping failed")
            
            result = pipeline.run_complete_pipeline(sample_urls)
            
            # Verify error was captured
            assert len(result.errors) > 0
            assert "Pipeline failed" in result.errors[0]
            assert result.total_urls_processed == 0
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_intermediate_results_saving(self, mock_qdrant, pipeline_config, temp_storage_dir, mock_vector_db):
        """Test intermediate results are saved when configured."""
        mock_qdrant.return_value = mock_vector_db
        
        # Enable intermediate results saving
        pipeline_config.save_intermediate_results = True
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Test saving intermediate results
        test_results = {'test_key': 'test_value', 'count': 42}
        pipeline._save_intermediate_results('test_stage', test_results)
        
        # Verify file was created
        results_file = temp_storage_dir / 'test_stage_results.json'
        assert results_file.exists()
        
        # Verify content
        with open(results_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['test_key'] == 'test_value'
        assert saved_data['count'] == 42
    
    @patch('src.database.integration_pipeline.QdrantVectorDB')
    def test_pipeline_status_tracking(self, mock_qdrant, pipeline_config, temp_storage_dir, mock_vector_db):
        """Test pipeline status tracking functionality."""
        mock_qdrant.return_value = mock_vector_db
        
        pipeline = IntegrationPipeline(
            config=pipeline_config,
            storage_dir=temp_storage_dir
        )
        
        # Get initial status
        status = pipeline.get_pipeline_status()
        assert status['current_stage'] is None
        assert status['start_time'] is None
        assert status['components_initialized'] is True
        
        # Simulate pipeline start
        pipeline.pipeline_state['start_time'] = time.time()
        pipeline.pipeline_state['current_stage'] = 'scraping'
        
        status = pipeline.get_pipeline_status()
        assert status['current_stage'] == 'scraping'
        assert status['start_time'] is not None
        assert status['elapsed_time'] >= 0
    
    def test_success_rate_calculation(self, pipeline_config, temp_storage_dir):
        """Test success rate calculation logic."""
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant:
            mock_qdrant.return_value.get_collections.return_value = []
            
            pipeline = IntegrationPipeline(
                config=pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            # Create test result
            result = PipelineResult(
                total_urls_processed=10,
                total_content_extracted=8,
                total_chunks_created=15,
                total_embeddings_generated=12,
                total_vectors_stored=10,
                processing_times={},
                success_rates={},
                errors=[]
            )
            
            success_rates = pipeline._calculate_success_rates(result)
            
            assert success_rates['scraping'] == 0.8  # 8/10
            assert success_rates['processing'] == 1.875  # 15/8 (chunks per content)
            assert success_rates['embedding'] == 0.8  # 12/15
            assert success_rates['ingestion'] == 10/12  # 10/12
            assert success_rates['overall'] == 1.0  # 10/10 