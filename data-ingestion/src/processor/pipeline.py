"""
Processing Pipeline Module

This module orchestrates the complete content processing pipeline,
integrating text processing, metadata extraction, and quality control.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

from .text_processor import TextProcessor, TextChunk, ChunkingConfig
from .metadata_extractor import MetadataExtractor, ExtractedMetadata
from .quality_control import QualityController, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for content processing pipeline."""
    # Text processing configuration
    chunking: ChunkingConfig
    
    # Quality control configuration
    min_quality_score: float = 0.6
    similarity_threshold: float = 0.85
    min_unique_words_ratio: float = 0.3
    
    # Processing configuration
    max_workers: int = 4
    batch_size: int = 50
    save_intermediate_results: bool = True
    
    # Output configuration
    output_dir: Path = Path("data/processed")
    include_metadata: bool = True
    include_quality_metrics: bool = True


@dataclass
class ProcessingResult:
    """Result of content processing pipeline."""
    total_pages: int
    total_chunks: int
    valid_chunks: int
    filtered_chunks: int
    duplicate_chunks: int
    processing_time_seconds: float
    output_files: List[str]
    quality_report: Dict[str, Any]
    errors: List[str]


class ProcessingPipeline:
    """
    Complete content processing pipeline.
    
    Orchestrates:
    - Text cleaning and chunking
    - Metadata extraction
    - Quality validation and filtering
    - Duplicate detection and removal
    - Result aggregation and storage
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the processing pipeline.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        
        # Initialize components
        self.text_processor = TextProcessor(config.chunking)
        self.metadata_extractor = MetadataExtractor()
        self.quality_controller = QualityController({
            'quality_threshold': config.min_quality_score,
            'similarity_threshold': config.similarity_threshold,
            'min_unique_words_ratio': config.min_unique_words_ratio
        })
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'pages_processed': 0,
            'chunks_created': 0,
            'chunks_filtered': 0,
            'duplicates_removed': 0,
            'errors': []
        }
        
        logger.info("ProcessingPipeline initialized")
    
    def process_scraped_data(self, 
                           scraped_data_file: Path,
                           progress_callback: Optional[Callable[[str, int, int], None]] = None) -> ProcessingResult:
        """
        Process scraped data from file.
        
        Args:
            scraped_data_file: Path to scraped data JSON file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        # Load scraped data
        try:
            with open(scraped_data_file, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
        except Exception as e:
            error_msg = f"Failed to load scraped data: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                total_pages=0,
                total_chunks=0,
                valid_chunks=0,
                filtered_chunks=0,
                duplicate_chunks=0,
                processing_time_seconds=0.0,
                output_files=[],
                quality_report={},
                errors=[error_msg]
            )
        
        # Extract pages from scraped data - handle multiple formats
        pages = []
        
        if isinstance(scraped_data, list):
            # Format: [{page1}, {page2}, ...] (from old orchestrator or enhanced as list)
            pages = scraped_data
            logger.info(f"Detected list format with {len(pages)} pages")
            
        elif isinstance(scraped_data, dict):
            # Format: {"pages": [...], ...} (expected format)
            if 'pages' in scraped_data:
                pages = scraped_data['pages']
                logger.info(f"Detected dict format with {len(pages)} pages")
            elif 'scraped_pages' in scraped_data:
                # Enhanced comprehensive format: {"scraped_pages": [...], ...}
                pages = scraped_data['scraped_pages']
                logger.info(f"Detected enhanced comprehensive format with {len(pages)} pages")
            else:
                # Try to treat the entire dict as a single page
                logger.warning("Unknown dict format, treating as single page")
                pages = [scraped_data]
        
        if not pages:
            error_msg = f"No pages found in scraped data. Data type: {type(scraped_data)}, Keys: {list(scraped_data.keys()) if isinstance(scraped_data, dict) else 'N/A'}"
            logger.error(error_msg)
            return ProcessingResult(
                total_pages=0,
                total_chunks=0,
                valid_chunks=0,
                filtered_chunks=0,
                duplicate_chunks=0,
                processing_time_seconds=0.0,
                output_files=[],
                quality_report={},
                errors=[error_msg]
            )
        
        # Process the pages
        return self.process_page_data(pages, progress_callback)
    
    def process_page_data(self, 
                         page_data_list: List[Dict[str, Any]],
                         progress_callback: Optional[Callable[[str, int, int], None]] = None) -> ProcessingResult:
        """
        Process list of page data.
        
        Args:
            page_data_list: List of scraped page data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results
        """
        start_time = time.time()
        logger.info(f"Starting processing of {len(page_data_list)} pages")
        
        # Process pages in batches
        all_chunks = []
        all_metadata = []
        all_quality_metrics = []
        errors = []
        
        # Create batches
        batches = [
            page_data_list[i:i + self.config.batch_size]
            for i in range(0, len(page_data_list), self.config.batch_size)
        ]
        
        # Process batches with parallel execution
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, batch_idx): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            
            completed_batches = 0
            for future in tqdm(as_completed(future_to_batch), 
                             total=len(batches), 
                             desc="Processing batches"):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_result = future.result()
                    all_chunks.extend(batch_result['chunks'])
                    all_metadata.extend(batch_result['metadata'])
                    all_quality_metrics.extend(batch_result['quality_metrics'])
                    
                    if batch_result['errors']:
                        errors.extend(batch_result['errors'])
                    
                except Exception as e:
                    error_msg = f"Batch {batch_idx} processing failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                
                completed_batches += 1
                if progress_callback:
                    progress_callback('processing', completed_batches, len(batches))
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(page_data_list)} pages")
        
        # Quality filtering
        if progress_callback:
            progress_callback('quality_filtering', 0, 1)
        
        filtered_chunks, quality_metrics = self.quality_controller.filter_quality_chunks(
            all_chunks
        )
        
        # Duplicate removal
        if progress_callback:
            progress_callback('duplicate_removal', 0, 1)
        
        deduplicated_chunks, duplicate_info = self.quality_controller.remove_duplicates(
            filtered_chunks
        )
        
        # Generate quality report
        quality_report = self.quality_controller.generate_quality_report(
            quality_metrics
        )
        
        # Save results
        output_files = []
        if self.config.save_intermediate_results:
            output_files = self._save_processing_results(
                deduplicated_chunks, all_metadata, quality_report, duplicate_info
            )
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            total_pages=len(page_data_list),
            total_chunks=len(all_chunks),
            valid_chunks=len(filtered_chunks),
            filtered_chunks=len(all_chunks) - len(filtered_chunks),
            duplicate_chunks=len(filtered_chunks) - len(deduplicated_chunks),
            processing_time_seconds=processing_time,
            output_files=output_files,
            quality_report=quality_report,
            errors=errors
        )
        
        logger.info(f"Processing completed: {len(deduplicated_chunks)} final chunks in {processing_time:.2f}s")
        return result
    
    def _process_batch(self, batch: List[Dict[str, Any]], batch_idx: int) -> Dict[str, Any]:
        """Process a batch of pages."""
        batch_chunks = []
        batch_metadata = []
        batch_quality_metrics = []
        batch_errors = []
        
        for page_data in batch:
            try:
                # Process page content into chunks
                chunks = self.text_processor.process_page_content(page_data)
                
                # Extract metadata for each chunk
                for chunk in chunks:
                    try:
                        # Prepare context for metadata extraction
                        context = {
                            'url': chunk.source_url,
                            'page_type': chunk.metadata.get('page_type', 'unknown'),
                            'heading': chunk.heading
                        }
                        
                        # Extract metadata
                        metadata = self.metadata_extractor.extract_metadata(
                            chunk.clean_text, context
                        )
                        
                        # Validate quality
                        quality = self.quality_controller.validate_chunk_quality(
                            chunk
                        )
                        
                        batch_chunks.append(chunk)
                        batch_metadata.append(metadata)
                        batch_quality_metrics.append(quality)
                        
                    except Exception as e:
                        error_msg = f"Error processing chunk from {page_data.get('url', 'unknown')}: {str(e)}"
                        batch_errors.append(error_msg)
                        logger.debug(error_msg)
                
            except Exception as e:
                error_msg = f"Error processing page {page_data.get('url', 'unknown')}: {str(e)}"
                batch_errors.append(error_msg)
                logger.debug(error_msg)
        
        return {
            'chunks': batch_chunks,
            'metadata': batch_metadata,
            'quality_metrics': batch_quality_metrics,
            'errors': batch_errors
        }
    
    def _save_processing_results(self, 
                               chunks: List[TextChunk],
                               metadata_list: List[ExtractedMetadata],
                               quality_report: Dict[str, Any],
                               duplicate_info: Dict[str, Any]) -> List[str]:
        """Save processing results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        try:
            # Save processed chunks
            chunks_file = self.config.output_dir / f"processed_chunks_{timestamp}.json"
            chunks_data = []
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': chunk.id,
                    'text': chunk.text,
                    'clean_text': chunk.clean_text,
                    'metadata': chunk.metadata,
                    'char_count': chunk.char_count,
                    'word_count': chunk.word_count,
                    'sentence_count': chunk.sentence_count,
                    'chunk_index': chunk.chunk_index,
                    'source_url': chunk.source_url,
                    'heading': chunk.heading,
                    'heading_level': chunk.heading_level
                }
                
                # Add extracted metadata if available
                if i < len(metadata_list):
                    chunk_data['extracted_metadata'] = asdict(metadata_list[i])
                
                chunks_data.append(chunk_data)
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            output_files.append(str(chunks_file))
            logger.info(f"Saved {len(chunks)} processed chunks to {chunks_file}")
            
            # Save quality report
            quality_file = self.config.output_dir / f"quality_report_{timestamp}.json"
            with open(quality_file, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)
            
            output_files.append(str(quality_file))
            
            # Save duplicate information
            if duplicate_info:
                duplicates_file = self.config.output_dir / f"duplicates_{timestamp}.json"
                # Convert DuplicateInfo objects to dictionaries
                serializable_duplicates = {}
                for original_id, duplicates in duplicate_info.items():
                    serializable_duplicates[original_id] = [
                        {
                            'chunk_id': dup.chunk_id,
                            'similarity_score': dup.similarity_score,
                            'duplicate_type': dup.duplicate_type
                        }
                        for dup in duplicates
                    ]
                
                with open(duplicates_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_duplicates, f, indent=2, ensure_ascii=False)
                
                output_files.append(str(duplicates_file))
            
            # Save processing summary
            summary_file = self.config.output_dir / f"processing_summary_{timestamp}.json"
            summary = {
                'timestamp': timestamp,
                'total_chunks': len(chunks),
                'output_files': output_files,
                'config': {
                    'chunking': asdict(self.config.chunking),
                    'min_quality_score': self.config.min_quality_score,
                    'similarity_threshold': self.config.similarity_threshold,
                    'batch_size': self.config.batch_size,
                    'max_workers': self.config.max_workers
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            output_files.append(str(summary_file))
            
        except Exception as e:
            logger.error(f"Error saving processing results: {str(e)}")
        
        return output_files
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'pages_processed': self.stats['pages_processed'],
            'chunks_created': self.stats['chunks_created'],
            'chunks_filtered': self.stats['chunks_filtered'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'error_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else []
        }
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'pages_processed': 0,
            'chunks_created': 0,
            'chunks_filtered': 0,
            'duplicates_removed': 0,
            'errors': []
        } 