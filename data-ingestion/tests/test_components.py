"""
Unit Tests for Individual Pipeline Components

Tests each component in isolation to ensure proper functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import json
from datetime import datetime

# Import components to test
from src.scraper import BaseScraper, UrlDiscovery, ContentExtractor
from src.processor import TextCleaner, TextChunker, MetadataExtractor, QualityControl
from src.embeddings import ModelManager, TextPreprocessor, EmbeddingGenerator, EmbeddingValidator
from src.database import QdrantVectorDB, VectorDBManager, SearchResult


class TestScrapingComponents:
    """Test scraping components."""
    
    def test_base_scraper_initialization(self):
        """Test BaseScraper initialization."""
        scraper = BaseScraper()
        assert scraper.session is not None
        assert scraper.max_retries == 3
        assert scraper.delay_range == (1, 3)
    
    def test_url_discovery_sitemap_parsing(self):
        """Test URL discovery from sitemap."""
        discovery = UrlDiscovery()
        
        # Mock sitemap content
        mock_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>"""
        
        with patch.object(discovery, '_fetch_sitemap') as mock_fetch:
            mock_fetch.return_value = mock_sitemap
            
            urls = discovery.discover_urls('https://example.com/sitemap.xml')
            
            assert len(urls) == 2
            assert 'https://example.com/page1' in urls
            assert 'https://example.com/page2' in urls
    
    def test_content_extractor_basic_extraction(self):
        """Test basic content extraction."""
        extractor = ContentExtractor()
        
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is test content.</p>
                <div class="content">Important information here.</div>
            </body>
        </html>
        """
        
        result = extractor.extract_content(html_content, 'https://example.com/test')
        
        assert result['title'] == 'Test Page'
        assert 'Main Heading' in result['content']
        assert 'test content' in result['content']
        assert 'Important information' in result['content']
        assert result['url'] == 'https://example.com/test'


class TestProcessingComponents:
    """Test content processing components."""
    
    def test_text_cleaner_unicode_normalization(self):
        """Test text cleaning and Unicode normalization."""
        cleaner = TextCleaner()
        
        # Test with various Unicode characters
        text = "Café naïve résumé 🚀 \u00A0 \u2013 \u201C"
        cleaned = cleaner.clean_text(text)
        
        assert 'Café' in cleaned
        assert 'naïve' in cleaned
        assert 'résumé' in cleaned
        assert '🚀' in cleaned
        # Non-breaking space should be normalized to regular space
        assert '\u00A0' not in cleaned
    
    def test_text_chunker_smart_splitting(self):
        """Test intelligent text chunking."""
        chunker = TextChunker(max_chunk_size=100, chunk_overlap=20)
        
        text = "This is the first sentence. " * 10 + "This is the second part. " * 10
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        # Check that chunks respect size limits
        for chunk in chunks:
            assert len(chunk) <= 120  # max_size + some tolerance
        
        # Check overlap exists between consecutive chunks
        if len(chunks) > 1:
            # Should have some overlap between chunks
            assert any(word in chunks[1] for word in chunks[0].split()[-5:])
    
    def test_metadata_extractor_keyword_extraction(self):
        """Test metadata extraction and keyword identification."""
        extractor = MetadataExtractor()
        
        text = "Machine learning and artificial intelligence are transforming business automation. " \
               "Companies use ML algorithms for data analysis and predictive analytics."
        
        metadata = extractor.extract_metadata(text)
        
        assert 'keywords' in metadata
        assert 'language' in metadata
        assert 'content_type' in metadata
        
        # Check for expected keywords
        keywords = [kw.lower() for kw in metadata['keywords']]
        assert any('machine' in kw or 'learning' in kw for kw in keywords)
        assert any('artificial' in kw or 'intelligence' in kw for kw in keywords)
    
    def test_quality_control_scoring(self):
        """Test content quality assessment."""
        qc = QualityControl()
        
        # High quality text
        good_text = "This is a well-written article about machine learning. " \
                   "It contains proper sentences with clear structure. " \
                   "The content is informative and provides valuable insights " \
                   "about artificial intelligence applications in business."
        
        # Low quality text
        poor_text = "txt msg lol omg wtf " * 10
        
        good_metrics = qc.assess_quality(good_text)
        poor_metrics = qc.assess_quality(poor_text)
        
        assert good_metrics.overall_score > poor_metrics.overall_score
        assert good_metrics.language_score > poor_metrics.language_score
        assert good_metrics.readability_score > poor_metrics.readability_score


class TestEmbeddingComponents:
    """Test embedding generation components."""
    
    def test_model_manager_model_loading(self):
        """Test model loading and management."""
        manager = ModelManager()
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            model = manager.load_model('all-MiniLM-L6-v2')
            
            assert model is not None
            mock_st.assert_called_once_with('all-MiniLM-L6-v2')
    
    def test_text_preprocessor_batch_processing(self):
        """Test text preprocessing in batches."""
        preprocessor = TextPreprocessor()
        
        texts = [
            "This is a test sentence with CAPS and numbers 123.",
            "Another sentence with   extra   spaces.",
            "A third sentence with special chars: @#$%"
        ]
        
        processed = preprocessor.preprocess_batch(texts)
        
        assert len(processed) == len(texts)
        for text in processed:
            assert isinstance(text, str)
            assert len(text) > 0
            # Should not have excessive whitespace
            assert '   ' not in text
    
    def test_embedding_generator_batch_generation(self):
        """Test embedding generation in batches."""
        generator = EmbeddingGenerator()
        
        # Mock model
        mock_model = Mock()
        mock_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = mock_embeddings
        
        texts = ["Text one", "Text two", "Text three"]
        
        result = generator.generate_embeddings(texts, mock_model, batch_size=2)
        
        assert 'embeddings' in result
        assert 'statistics' in result
        assert len(result['embeddings']) == 3
        assert result['embeddings'][0].shape == (384,)
        
        # Verify model was called
        mock_model.encode.assert_called()
    
    def test_embedding_validator_dimension_consistency(self):
        """Test embedding validation for dimension consistency."""
        validator = EmbeddingValidator(expected_dimensions=384)
        
        # Consistent embeddings
        good_embeddings = [np.random.rand(384) for _ in range(5)]
        
        # Inconsistent embeddings
        bad_embeddings = [
            np.random.rand(384),
            np.random.rand(512),  # Wrong dimension
            np.random.rand(384)
        ]
        
        good_result = validator.validate_embeddings(good_embeddings)
        bad_result = validator.validate_embeddings(bad_embeddings)
        
        assert good_result['overall_valid'] is True
        assert bad_result['overall_valid'] is False
        assert 'dimension_mismatch' in str(bad_result['issues'])


class TestDatabaseComponents:
    """Test database components."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock basic operations
            mock_instance.get_collections.return_value = Mock(collections=[])
            mock_instance.create_collection.return_value = True
            mock_instance.upsert.return_value = Mock(operation_id='test', status='completed')
            
            yield mock_instance
    
    def test_qdrant_vector_db_initialization(self, mock_qdrant_client):
        """Test QdrantVectorDB initialization."""
        db = QdrantVectorDB(host='localhost', port=6333)
        
        assert db.host == 'localhost'
        assert db.port == 6333
        assert db.client is not None
    
    def test_vector_db_collection_operations(self, mock_qdrant_client):
        """Test collection create/delete operations."""
        db = QdrantVectorDB()
        
        # Test collection creation
        success = db.create_collection('test_collection', vector_size=384)
        assert success is True
        
        # Test collection existence check
        mock_qdrant_client.get_collections.return_value = Mock(
            collections=[Mock(name='test_collection')]
        )
        exists = db.collection_exists('test_collection')
        assert exists is True
    
    def test_vector_insertion(self, mock_qdrant_client):
        """Test vector insertion with metadata."""
        db = QdrantVectorDB()
        
        # Mock collection exists
        mock_qdrant_client.get_collections.return_value = Mock(
            collections=[Mock(name='test_collection')]
        )
        
        vectors = [np.random.rand(384) for _ in range(3)]
        texts = ['Text 1', 'Text 2', 'Text 3']
        metadata = [{'source': 'test'} for _ in range(3)]
        
        result = db.insert_vectors(
            collection_name='test_collection',
            vectors=vectors,
            texts=texts,
            metadata=metadata
        )
        
        assert result['inserted_count'] == 3
        assert result['collection_name'] == 'test_collection'
        
        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
    
    def test_vector_search(self, mock_qdrant_client):
        """Test vector similarity search."""
        db = QdrantVectorDB()
        
        # Mock collection exists
        mock_qdrant_client.get_collections.return_value = Mock(
            collections=[Mock(name='test_collection')]
        )
        
        # Mock search results
        mock_results = [
            Mock(
                id='1',
                score=0.95,
                payload={'text': 'Sample text 1', 'metadata': 'value1'}
            ),
            Mock(
                id='2',
                score=0.87,
                payload={'text': 'Sample text 2', 'metadata': 'value2'}
            )
        ]
        mock_qdrant_client.search.return_value = mock_results
        
        query_vector = np.random.rand(384)
        results = db.search(
            collection_name='test_collection',
            query_vector=query_vector,
            limit=10
        )
        
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].text == 'Sample text 1'
        assert results[1].score == 0.87
        
        # Verify search was called
        mock_qdrant_client.search.assert_called_once()
    
    def test_vector_db_manager_ingestion(self, mock_qdrant_client):
        """Test high-level vector database manager."""
        db = QdrantVectorDB()
        manager = VectorDBManager(db)
        
        # Mock successful operations
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        
        # Create mock embedding records
        embedding_records = []
        for i in range(5):
            record = Mock()
            record.id = f'record_{i}'
            record.text = f'Sample text {i}'
            record.embedding = np.random.rand(384)
            record.metadata = {'index': i}
            record.embedding_dimensions = 384
            embedding_records.append(record)
        
        from src.database.vector_db_manager import IngestionConfig
        config = IngestionConfig(
            collection_name='test_collection',
            vector_size=384,
            batch_size=2
        )
        
        # Mock the batch processing
        with patch.object(manager, '_process_batch') as mock_process:
            mock_process.return_value = {'inserted_count': 2}
            
            result = manager.ingest_embeddings(embedding_records, config)
            
            assert result.total_processed == 5
            assert result.successful_insertions >= 0
            assert isinstance(result.processing_time_seconds, float)


class TestDataValidation:
    """Test data validation and quality assurance."""
    
    def test_embedding_record_validation(self):
        """Test EmbeddingRecord data validation."""
        from src.embeddings import EmbeddingRecord
        
        # Valid record
        valid_record = EmbeddingRecord(
            id='test_id',
            text='Test text content',
            embedding=np.random.rand(384),
            metadata={'source': 'test'},
            timestamp=datetime.now().isoformat(),
            source='test',
            embedding_model='test-model',
            embedding_dimensions=384
        )
        
        assert valid_record.id == 'test_id'
        assert valid_record.embedding_dimensions == 384
        assert len(valid_record.embedding) == 384
        
        # Test dictionary conversion
        record_dict = valid_record.to_dict()
        assert 'id' in record_dict
        assert 'embedding' in record_dict
        assert isinstance(record_dict['embedding'], list)
    
    def test_search_result_validation(self):
        """Test SearchResult data validation."""
        result = SearchResult(
            id='result_1',
            score=0.95,
            text='Sample result text',
            metadata={'category': 'test'}
        )
        
        assert result.id == 'result_1'
        assert result.score == 0.95
        assert 'Sample result' in result.text
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        assert result_dict['score'] == 0.95
        assert result_dict['metadata']['category'] == 'test'
    
    def test_pipeline_config_validation(self):
        """Test pipeline configuration validation."""
        from src.database import PipelineConfig
        from src.scraper import ScrapingConfig
        from src.processor import ProcessingConfig
        
        config = PipelineConfig(
            scraping=ScrapingConfig(
                base_url="https://example.com",
                max_pages=10
            ),
            processing=ProcessingConfig(
                min_chunk_size=50,
                max_chunk_size=1000
            ),
            embedding_model="test-model",
            collection_name="test_collection"
        )
        
        assert config.embedding_model == "test-model"
        assert config.collection_name == "test_collection"
        assert config.scraping.max_pages == 10
        assert config.processing.min_chunk_size == 50


class TestErrorHandling:
    """Test error handling across components."""
    
    def test_scraper_error_handling(self):
        """Test scraper error handling."""
        scraper = BaseScraper()
        
        # Test with invalid URL
        with patch.object(scraper.session, 'get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            result = scraper.scrape_url('https://invalid-url.com')
            
            # Should handle error gracefully
            assert result is not None
            assert 'error' in result or result == {}
    
    def test_embedding_generation_error_handling(self):
        """Test embedding generation error handling."""
        generator = EmbeddingGenerator()
        
        # Mock model that fails
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Model error")
        
        texts = ["Test text"]
        
        # Should handle model errors gracefully
        with pytest.raises(Exception):
            generator.generate_embeddings(texts, mock_model)
    
    def test_database_connection_error_handling(self):
        """Test database connection error handling."""
        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            
            # Should raise exception on connection failure
            with pytest.raises(Exception):
                QdrantVectorDB(host='invalid-host', port=9999)


class TestPerformanceMetrics:
    """Test performance monitoring and metrics."""
    
    def test_processing_time_tracking(self):
        """Test processing time tracking in components."""
        from src.processor import ProcessingPipeline, ProcessingConfig
        
        config = ProcessingConfig()
        pipeline = ProcessingPipeline(config)
        
        # Mock the internal components
        with patch.object(pipeline.text_cleaner, 'clean_text') as mock_clean, \
             patch.object(pipeline.text_chunker, 'chunk_text') as mock_chunk, \
             patch.object(pipeline.metadata_extractor, 'extract_metadata') as mock_meta, \
             patch.object(pipeline.quality_control, 'assess_quality') as mock_quality:
            
            mock_clean.return_value = "cleaned text"
            mock_chunk.return_value = ["chunk1", "chunk2"]
            mock_meta.return_value = {"keywords": ["test"]}
            mock_quality.return_value = Mock(overall_score=0.8)
            
            result = pipeline.process_content("test content")
            
            assert 'processing_time' in result
            assert isinstance(result['processing_time'], float)
            assert result['processing_time'] >= 0
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during processing."""
        # This would require actual memory profiling in a real scenario
        # For now, we'll test that components can handle large inputs
        
        large_text = "This is a test sentence. " * 1000
        
        from src.processor import TextCleaner
        cleaner = TextCleaner()
        
        # Should handle large inputs without memory issues
        result = cleaner.clean_text(large_text)
        assert len(result) > 0
        assert isinstance(result, str) 