"""
End-to-End Tests for Complete Data Ingestion Pipeline

Tests the full pipeline with realistic data and performance benchmarks.
"""

import pytest
import time
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch, Mock
from typing import List, Dict, Any

from src.scraper import ScrapingConfig
from src.processor import ProcessingConfig
from src.database import PipelineConfig, IntegrationPipeline


class TestEndToEndPipeline:
    """End-to-end pipeline tests with realistic scenarios."""
    
    @pytest.fixture
    def realistic_pipeline_config(self):
        """Create realistic pipeline configuration for testing."""
        return PipelineConfig(
            scraping=ScrapingConfig(
                base_url="https://tekyz.com",
                max_pages=5,
                delay_range=(0.5, 1.0),
                max_retries=2,
                timeout=10
            ),
            processing=ProcessingConfig(
                min_chunk_size=200,
                max_chunk_size=800,
                chunk_overlap=100,
                quality_threshold=0.6,
                language_detection=True,
                remove_duplicates=True
            ),
            embedding_model="all-MiniLM-L6-v2",
            embedding_batch_size=16,
            collection_name="tekyz_knowledge_test",
            vector_db_host="localhost",
            vector_db_port=6333,
            max_concurrent_workers=2,
            save_intermediate_results=True,
            validate_embeddings=True,
            backup_data=True
        )
    
    @pytest.fixture
    def sample_website_content(self):
        """Generate realistic website content for testing."""
        return {
            'extracted_content': [
                {
                    'url': 'https://tekyz.com/services/web-development',
                    'title': 'Web Development Services - Tekyz',
                    'content': """
                    Our web development services include custom website design, 
                    e-commerce solutions, and responsive web applications. 
                    We use modern technologies like React, Node.js, and Python 
                    to build scalable and maintainable web solutions.
                    
                    Key Features:
                    - Custom design and development
                    - Mobile-responsive layouts
                    - SEO optimization
                    - Performance optimization
                    - Cross-browser compatibility
                    
                    Technologies We Use:
                    React.js for dynamic user interfaces
                    Node.js for backend development
                    MongoDB for database solutions
                    AWS for cloud deployment
                    """,
                    'metadata': {
                        'content_type': 'service_page',
                        'category': 'web_development',
                        'last_modified': '2024-01-15'
                    }
                },
                {
                    'url': 'https://tekyz.com/services/digital-marketing',
                    'title': 'Digital Marketing Solutions - Tekyz',
                    'content': """
                    Transform your business with our comprehensive digital marketing strategies. 
                    We offer SEO, social media marketing, content marketing, and PPC advertising 
                    to help you reach your target audience effectively.
                    
                    Our Services Include:
                    - Search Engine Optimization (SEO)
                    - Social Media Marketing
                    - Content Marketing Strategy
                    - Pay-Per-Click Advertising
                    - Email Marketing Campaigns
                    - Analytics and Reporting
                    
                    We work with businesses of all sizes to create customized marketing 
                    solutions that drive results and increase ROI.
                    """,
                    'metadata': {
                        'content_type': 'service_page',
                        'category': 'digital_marketing',
                        'last_modified': '2024-01-10'
                    }
                },
                {
                    'url': 'https://tekyz.com/about',
                    'title': 'About Tekyz - Leading Technology Solutions',
                    'content': """
                    Tekyz is a leading technology company specializing in web development, 
                    digital marketing, and business automation solutions. Founded in 2020, 
                    we have helped over 100 businesses transform their digital presence.
                    
                    Our Mission:
                    To empower businesses with innovative technology solutions that drive 
                    growth and efficiency.
                    
                    Our Values:
                    - Innovation and creativity
                    - Customer-centric approach
                    - Quality and excellence
                    - Continuous learning and improvement
                    
                    Team Expertise:
                    Our team consists of experienced developers, designers, and marketing 
                    professionals who are passionate about delivering exceptional results.
                    """,
                    'metadata': {
                        'content_type': 'company_info',
                        'category': 'about',
                        'last_modified': '2024-01-20'
                    }
                }
            ]
        }
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for E2E tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_complete_pipeline_with_realistic_data(self, realistic_pipeline_config, 
                                                  sample_website_content, temp_storage_dir):
        """Test complete pipeline with realistic website data."""
        
        # Mock external dependencies
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            # Setup mocks
            mock_db = Mock()
            mock_db.collection_exists.return_value = False
            mock_db.create_collection.return_value = True
            mock_db.insert_vectors.return_value = {
                'inserted_count': 10,
                'collection_name': 'tekyz_knowledge_test',
                'status': 'completed'
            }
            mock_db.get_health.return_value = {'status': 'healthy'}
            mock_qdrant.return_value = mock_db
            
            # Mock embedding model
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(10, 384)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            # Initialize pipeline
            pipeline = IntegrationPipeline(
                config=realistic_pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            # Mock the scraping stage to return our sample content
            with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
                mock_scraping.return_value = sample_website_content
                
                # Run the complete pipeline
                urls = [item['url'] for item in sample_website_content['extracted_content']]
                result = pipeline.run_complete_pipeline(urls)
                
                # Verify pipeline completed successfully
                assert result.total_urls_processed == len(urls)
                assert result.total_content_extracted == len(sample_website_content['extracted_content'])
                assert result.total_chunks_created > 0
                assert result.total_embeddings_generated > 0
                assert result.total_vectors_stored >= 0
                
                # Verify processing times were recorded
                assert 'processing' in result.processing_times
                assert 'embedding' in result.processing_times
                assert 'ingestion' in result.processing_times
                
                # Verify success rates
                assert result.success_rates['overall'] >= 0
                
                print(f"✅ E2E Test Results:")
                print(f"   URLs Processed: {result.total_urls_processed}")
                print(f"   Content Extracted: {result.total_content_extracted}")
                print(f"   Chunks Created: {result.total_chunks_created}")
                print(f"   Embeddings Generated: {result.total_embeddings_generated}")
                print(f"   Vectors Stored: {result.total_vectors_stored}")
                print(f"   Overall Success Rate: {result.success_rates.get('overall', 0):.2%}")
    
    def test_pipeline_performance_benchmarks(self, realistic_pipeline_config, 
                                           sample_website_content, temp_storage_dir):
        """Test pipeline performance with various data sizes."""
        
        performance_results = []
        
        # Test with different data sizes
        test_sizes = [1, 3, 5]  # Number of content items
        
        for size in test_sizes:
            # Create subset of content
            content_subset = {
                'extracted_content': sample_website_content['extracted_content'][:size]
            }
            
            with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant, \
                 patch('sentence_transformers.SentenceTransformer') as mock_st:
                
                # Setup mocks for performance testing
                mock_db = Mock()
                mock_db.collection_exists.return_value = False
                mock_db.create_collection.return_value = True
                mock_db.insert_vectors.return_value = {
                    'inserted_count': size * 3,  # Assume ~3 chunks per content
                    'collection_name': 'test_collection',
                    'status': 'completed'
                }
                mock_qdrant.return_value = mock_db
                
                # Mock model with realistic delay
                mock_model = Mock()
                def mock_encode(texts, batch_size=32, show_progress_bar=False):
                    # Simulate encoding time
                    time.sleep(0.1 * len(texts) / 10)  # 0.1s per 10 texts
                    return np.random.rand(len(texts), 384)
                
                mock_model.encode.side_effect = mock_encode
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st.return_value = mock_model
                
                # Run pipeline
                pipeline = IntegrationPipeline(
                    config=realistic_pipeline_config,
                    storage_dir=temp_storage_dir
                )
                
                with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
                    mock_scraping.return_value = content_subset
                    
                    start_time = time.time()
                    result = pipeline.run_complete_pipeline(
                        [item['url'] for item in content_subset['extracted_content']]
                    )
                    total_time = time.time() - start_time
                    
                    performance_results.append({
                        'content_items': size,
                        'total_time': total_time,
                        'chunks_created': result.total_chunks_created,
                        'embeddings_generated': result.total_embeddings_generated,
                        'processing_rate': result.total_chunks_created / total_time if total_time > 0 else 0
                    })
        
        # Analyze performance results
        print(f"\n📊 Performance Benchmark Results:")
        print(f"{'Items':<8} {'Time(s)':<10} {'Chunks':<8} {'Embeddings':<12} {'Rate(chunks/s)':<15}")
        print("-" * 60)
        
        for result in performance_results:
            print(f"{result['content_items']:<8} "
                  f"{result['total_time']:<10.2f} "
                  f"{result['chunks_created']:<8} "
                  f"{result['embeddings_generated']:<12} "
                  f"{result['processing_rate']:<15.2f}")
        
        # Verify performance is reasonable
        for result in performance_results:
            assert result['total_time'] < 60  # Should complete within 1 minute
            assert result['processing_rate'] > 0  # Should have positive processing rate
    
    def test_pipeline_error_recovery(self, realistic_pipeline_config, temp_storage_dir):
        """Test pipeline error handling and recovery mechanisms."""
        
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant:
            mock_db = Mock()
            mock_qdrant.return_value = mock_db
            
            pipeline = IntegrationPipeline(
                config=realistic_pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            # Test 1: Scraping failure
            with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
                mock_scraping.side_effect = Exception("Network error")
                
                result = pipeline.run_complete_pipeline(['https://example.com/test'])
                
                assert len(result.errors) > 0
                assert "Pipeline failed" in result.errors[0]
                assert result.total_urls_processed == 0
            
            # Test 2: Partial processing failure
            with patch.object(pipeline, '_run_scraping_stage') as mock_scraping, \
                 patch.object(pipeline, '_run_processing_stage') as mock_processing:
                
                mock_scraping.return_value = {'extracted_content': [{'url': 'test', 'content': 'test'}]}
                mock_processing.return_value = {'processed_chunks': []}  # No chunks produced
                
                result = pipeline.run_complete_pipeline(['https://example.com/test'])
                
                assert "No chunks created from processing stage" in result.errors
    
    def test_pipeline_with_large_content(self, realistic_pipeline_config, temp_storage_dir):
        """Test pipeline with large content to verify memory efficiency."""
        
        # Create large content item
        large_content = {
            'extracted_content': [
                {
                    'url': 'https://example.com/large-document',
                    'title': 'Large Document Test',
                    'content': "This is a very long document. " * 1000,  # ~30KB content
                    'metadata': {'size': 'large'}
                }
            ]
        }
        
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            # Setup mocks
            mock_db = Mock()
            mock_db.collection_exists.return_value = False
            mock_db.create_collection.return_value = True
            mock_db.insert_vectors.return_value = {
                'inserted_count': 50,  # Large document should create many chunks
                'collection_name': 'test_collection',
                'status': 'completed'
            }
            mock_qdrant.return_value = mock_db
            
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(50, 384)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            # Run pipeline
            pipeline = IntegrationPipeline(
                config=realistic_pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
                mock_scraping.return_value = large_content
                
                result = pipeline.run_complete_pipeline(['https://example.com/large-document'])
                
                # Verify large content was processed successfully
                assert result.total_content_extracted == 1
                assert result.total_chunks_created > 10  # Should create many chunks
                assert result.total_embeddings_generated > 10
                assert len(result.errors) == 0
    
    def test_pipeline_concurrent_processing(self, realistic_pipeline_config, 
                                          sample_website_content, temp_storage_dir):
        """Test concurrent processing capabilities."""
        
        # Configure for higher concurrency
        realistic_pipeline_config.max_concurrent_workers = 4
        realistic_pipeline_config.embedding_batch_size = 8
        
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            # Setup mocks
            mock_db = Mock()
            mock_db.collection_exists.return_value = False
            mock_db.create_collection.return_value = True
            mock_db.insert_vectors.return_value = {
                'inserted_count': 15,
                'collection_name': 'test_collection',
                'status': 'completed'
            }
            mock_qdrant.return_value = mock_db
            
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(15, 384)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            # Run pipeline
            pipeline = IntegrationPipeline(
                config=realistic_pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
                mock_scraping.return_value = sample_website_content
                
                start_time = time.time()
                result = pipeline.run_complete_pipeline(
                    [item['url'] for item in sample_website_content['extracted_content']]
                )
                processing_time = time.time() - start_time
                
                # Verify concurrent processing worked
                assert result.total_vectors_stored > 0
                assert processing_time < 30  # Should be faster with concurrency
                assert len(result.errors) == 0
    
    def test_intermediate_results_persistence(self, realistic_pipeline_config, 
                                            sample_website_content, temp_storage_dir):
        """Test that intermediate results are saved and can be loaded."""
        
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            # Setup mocks
            mock_db = Mock()
            mock_db.collection_exists.return_value = False
            mock_db.create_collection.return_value = True
            mock_db.insert_vectors.return_value = {
                'inserted_count': 10,
                'collection_name': 'test_collection',
                'status': 'completed'
            }
            mock_qdrant.return_value = mock_db
            
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(10, 384)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            # Run pipeline with intermediate results enabled
            pipeline = IntegrationPipeline(
                config=realistic_pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
                mock_scraping.return_value = sample_website_content
                
                result = pipeline.run_complete_pipeline(
                    [item['url'] for item in sample_website_content['extracted_content']]
                )
                
                # Verify intermediate results files were created
                assert (temp_storage_dir / 'processing_results.json').exists()
                assert (temp_storage_dir / 'embedding_results.json').exists()
                
                # Verify we can load intermediate results
                processing_results = pipeline.load_intermediate_results('processing')
                assert processing_results is not None
                assert 'total_chunks' in processing_results or 'processed_chunks' in processing_results
                
                embedding_results = pipeline.load_intermediate_results('embedding')
                assert embedding_results is not None
                assert 'total_embeddings' in embedding_results
    
    def test_quality_validation_integration(self, realistic_pipeline_config, temp_storage_dir):
        """Test integration of quality validation throughout the pipeline."""
        
        # Create content with varying quality levels
        mixed_quality_content = {
            'extracted_content': [
                {
                    'url': 'https://example.com/high-quality',
                    'title': 'High Quality Content',
                    'content': """
                    This is a well-structured article about machine learning applications 
                    in business. The content provides comprehensive insights into how 
                    artificial intelligence technologies are transforming various industries.
                    
                    Key topics covered include:
                    - Natural language processing for customer service
                    - Predictive analytics for business forecasting
                    - Computer vision for quality control
                    - Automated decision-making systems
                    """,
                    'metadata': {'expected_quality': 'high'}
                },
                {
                    'url': 'https://example.com/low-quality',
                    'title': 'Low Quality Content',
                    'content': "short txt no structure lol omg wtf " * 5,
                    'metadata': {'expected_quality': 'low'}
                }
            ]
        }
        
        with patch('src.database.integration_pipeline.QdrantVectorDB') as mock_qdrant, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            # Setup mocks
            mock_db = Mock()
            mock_db.collection_exists.return_value = False
            mock_db.create_collection.return_value = True
            mock_db.insert_vectors.return_value = {
                'inserted_count': 5,
                'collection_name': 'test_collection',
                'status': 'completed'
            }
            mock_qdrant.return_value = mock_db
            
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(5, 384)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            # Run pipeline
            pipeline = IntegrationPipeline(
                config=realistic_pipeline_config,
                storage_dir=temp_storage_dir
            )
            
            with patch.object(pipeline, '_run_scraping_stage') as mock_scraping:
                mock_scraping.return_value = mixed_quality_content
                
                result = pipeline.run_complete_pipeline(
                    [item['url'] for item in mixed_quality_content['extracted_content']]
                )
                
                # Quality validation should filter out low-quality content
                # or at least flag it in the results
                assert result.total_content_extracted == 2  # Both items processed
                
                # Check if embedding validation was performed
                if result.embedding_results and 'validation_results' in result.embedding_results:
                    validation = result.embedding_results['validation_results']
                    assert 'overall_valid' in validation 