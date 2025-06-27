"""
Text Processing Module

This module handles text cleaning, normalization, and chunking for the data ingestion pipeline.
Implements advanced text processing techniques for optimal embedding generation.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

logger = logging.getLogger(__name__)

# Download required NLTK data
import os
nltk_data_path = os.environ.get('NLTK_DATA', None)
if nltk_data_path:
    nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK punkt data: {e}")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK stopwords data: {e}")


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    max_chunk_size: int = 800
    min_chunk_size: int = 50  # Reduced from 100 to 50
    overlap_size: int = 100
    split_by_sentences: bool = True
    preserve_paragraphs: bool = True
    respect_word_boundaries: bool = True


@dataclass
class TextChunk:
    """Represents a processed text chunk."""
    id: str
    text: str
    clean_text: str
    metadata: Dict[str, Any]
    char_count: int
    word_count: int
    sentence_count: int
    chunk_index: int
    source_url: str
    heading: Optional[str] = None
    heading_level: Optional[int] = None


class TextProcessor:
    """
    Advanced text processing for content preparation.
    
    Features:
    - HTML artifact removal
    - Unicode normalization
    - Text cleaning and normalization
    - Smart chunking with overlap
    - Quality validation
    """
    
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        """
        Initialize the text processor.
        
        Args:
            chunking_config: Configuration for text chunking
        """
        self.chunking_config = chunking_config or ChunkingConfig()
        
        # Initialize stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            self.stopwords = set()
            logger.warning("English stopwords not available")
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("TextProcessor initialized")
    
    def _compile_patterns(self):
        """Compile frequently used regex patterns."""
        # HTML artifacts
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.html_entity_pattern = re.compile(r'&[a-zA-Z0-9#]+;')
        
        # Whitespace normalization
        self.whitespace_pattern = re.compile(r'\s+')
        self.newline_pattern = re.compile(r'\n\s*\n\s*\n+')
        
        # URL and email patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone number pattern
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        
        # Quote normalization
        self.quote_patterns = [
            (re.compile(r'["""]'), '"'),
            (re.compile(r"[''']"), "'"),
        ]
        
        # Excessive punctuation
        self.punct_patterns = [
            (re.compile(r'[.]{3,}'), '...'),
            (re.compile(r'[-]{3,}'), '---'),
            (re.compile(r'[!]{2,}'), '!'),
            (re.compile(r'[?]{2,}'), '?'),
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags and entities
        text = self.html_tag_pattern.sub(' ', text)
        text = self.html_entity_pattern.sub(' ', text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs and emails (but keep meaningful content)
        text = self.url_pattern.sub(' [URL] ', text)
        text = self.email_pattern.sub(' [EMAIL] ', text)
        text = self.phone_pattern.sub(' [PHONE] ', text)
        
        # Normalize quotes
        for pattern, replacement in self.quote_patterns:
            text = pattern.sub(replacement, text)
        
        # Fix excessive punctuation
        for pattern, replacement in self.punct_patterns:
            text = pattern.sub(replacement, text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        text = self.newline_pattern.sub('\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Additional text normalization for consistency.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase for certain operations (preserve original case)
        # Remove excessive spacing
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common formatting issues
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces after newlines
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces before newlines
        
        # Fix punctuation spacing
        text = re.sub(r' +([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
        
        return text.strip()
    
    def create_chunks(self, 
                     text: str, 
                     source_url: str,
                     metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Create text chunks from processed content.
        
        Args:
            text: Clean text to chunk
            source_url: Source URL for the text
            metadata: Additional metadata for chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) < self.chunking_config.min_chunk_size:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Split into paragraphs first if configured
        if self.chunking_config.preserve_paragraphs:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        else:
            paragraphs = [text]
        
        chunk_index = 0
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph alone exceeds max size, split it
            if len(paragraph) > self.chunking_config.max_chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, chunk_index, source_url, metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = ""
                
                # Split large paragraph
                sub_chunks = self._split_large_text(paragraph)
                for sub_chunk in sub_chunks:
                    chunk = self._create_chunk(
                        sub_chunk, chunk_index, source_url, metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            # Check if adding paragraph would exceed max size
            elif len(current_chunk) + len(paragraph) + 2 > self.chunking_config.max_chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, chunk_index, source_url, metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap if configured
                if self.chunking_config.overlap_size > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.chunking_config.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk, chunk_index, source_url, metadata
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from text of {len(text)} characters")
        return chunks
    
    def _split_large_text(self, text: str) -> List[str]:
        """Split large text into smaller chunks."""
        if self.chunking_config.split_by_sentences:
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > self.chunking_config.max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        
                        # Add overlap
                        if self.chunking_config.overlap_size > 0:
                            overlap = self._get_overlap_text(current_chunk)
                            current_chunk = overlap + " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        # Single sentence is too long, split by words
                        word_chunks = self._split_by_words(sentence)
                        chunks.extend(word_chunks[:-1])
                        current_chunk = word_chunks[-1] if word_chunks else ""
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        else:
            # Split by character count with word boundaries
            return self._split_by_words(text)
    
    def _split_by_words(self, text: str) -> List[str]:
        """Split text by words while respecting boundaries."""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > self.chunking_config.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap
                    if self.chunking_config.overlap_size > 0:
                        overlap_words = current_chunk.split()[-10:]  # Last 10 words
                        current_chunk = " ".join(overlap_words) + " " + word
                    else:
                        current_chunk = word
                else:
                    # Single word is too long, truncate
                    chunks.append(word[:self.chunking_config.max_chunk_size])
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunking_config.overlap_size:
            return text
        
        # Try to get overlap by sentences
        sentences = sent_tokenize(text)
        overlap = ""
        
        for sentence in reversed(sentences):
            if len(overlap) + len(sentence) + 1 <= self.chunking_config.overlap_size:
                if overlap:
                    overlap = sentence + " " + overlap
                else:
                    overlap = sentence
            else:
                break
        
        # If no sentences fit, use character-based overlap
        if not overlap:
            overlap = text[-self.chunking_config.overlap_size:]
            # Try to start at word boundary
            space_idx = overlap.find(' ')
            if space_idx > 0:
                overlap = overlap[space_idx + 1:]
        
        return overlap
    
    def _create_chunk(self, 
                     text: str, 
                     chunk_index: int, 
                     source_url: str,
                     metadata: Dict[str, Any]) -> TextChunk:
        """Create a TextChunk object."""
        clean_text = self.normalize_text(text)
        
        # Calculate metrics
        char_count = len(clean_text)
        word_count = len(clean_text.split())
        sentence_count = len(sent_tokenize(clean_text))
        
        # Generate chunk ID
        chunk_id = f"{metadata.get('page_id', 'unknown')}_{chunk_index}"
        
        return TextChunk(
            id=chunk_id,
            text=text,
            clean_text=clean_text,
            metadata=metadata.copy(),
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            chunk_index=chunk_index,
            source_url=source_url,
            heading=metadata.get('heading'),
            heading_level=metadata.get('heading_level')
        )
    
    def validate_chunk_quality(self, chunk: TextChunk) -> Dict[str, Any]:
        """
        Validate the quality of a text chunk.
        
        Args:
            chunk: Text chunk to validate
            
        Returns:
            Quality metrics and validation results
        """
        quality_metrics = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0,
            'metrics': {}
        }
        
        # Check minimum length (more lenient)
        if chunk.char_count < self.chunking_config.min_chunk_size:
            quality_metrics['is_valid'] = False
            quality_metrics['issues'].append('Below minimum length')
            quality_metrics['quality_score'] -= 0.2
        
        # Check for excessive repetition (more lenient)
        words = chunk.clean_text.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            
            if repetition_ratio < 0.2:  # Less than 20% unique words (more lenient)
                quality_metrics['issues'].append('High repetition detected')
                quality_metrics['quality_score'] -= 0.1
            
            quality_metrics['metrics']['repetition_ratio'] = repetition_ratio
        
        # Check for meaningful content (more lenient)
        alpha_chars = sum(1 for c in chunk.clean_text if c.isalpha())
        alpha_ratio = alpha_chars / max(chunk.char_count, 1)
        
        if alpha_ratio < 0.3:  # Less than 30% alphabetic characters (more lenient)
            quality_metrics['issues'].append('Low alphabetic content')
            quality_metrics['quality_score'] -= 0.1
        
        quality_metrics['metrics']['alpha_ratio'] = alpha_ratio
        
        # Check sentence structure
        if chunk.sentence_count == 0:
            quality_metrics['issues'].append('No sentences detected')
            quality_metrics['quality_score'] -= 0.1
        
        avg_sentence_length = chunk.word_count / max(chunk.sentence_count, 1)
        if avg_sentence_length > 50:  # Very long sentences
            quality_metrics['issues'].append('Very long sentences')
            quality_metrics['quality_score'] -= 0.1
        
        quality_metrics['metrics']['avg_sentence_length'] = avg_sentence_length
        
        # Final quality score adjustment
        quality_metrics['quality_score'] = max(0.0, quality_metrics['quality_score'])
        
        return quality_metrics
    
    def process_page_content(self, 
                           page_data: Dict[str, Any]) -> List[TextChunk]:
        """
        Process complete page content into chunks.
        
        Args:
            page_data: Page data from scraper
            
        Returns:
            List of processed text chunks
        """
        # Extract text content - handle both old and new scraped data formats
        raw_text = ""
        
        # Try new format first (direct fields)
        if 'text' in page_data:
            raw_text = page_data.get('text', '')
        # Check if content is a string (uploaded files format)
        elif 'content' in page_data and isinstance(page_data['content'], str):
            raw_text = page_data.get('content', '')
        # Fallback to old format (nested content)
        elif 'content' in page_data:
            raw_text = page_data.get('content', {}).get('text', '')
        
        if not raw_text:
            logger.warning(f"No text content found for {page_data.get('url', page_data.get('metadata', {}).get('url', 'unknown'))}")
            return []
        
        # Clean the text
        clean_text = self.clean_text(raw_text)
        if not clean_text:
            logger.warning(f"No clean text after processing for {page_data.get('url', page_data.get('metadata', {}).get('url', 'unknown'))}")
            return []
        
        # Prepare metadata - handle both formats
        url = page_data.get('url') or page_data.get('metadata', {}).get('url', '')
        metadata_dict = page_data.get('metadata', {})
        
        metadata = {
            'page_id': url.replace('https://', '').replace('/', '_') if url else 'unknown',
            'page_title': metadata_dict.get('title', '') or page_data.get('title', '') or (page_data.get('content', {}).get('title', '') if isinstance(page_data.get('content'), dict) else ''),
            'page_type': metadata_dict.get('page_type', 'unknown') or (page_data.get('content', {}).get('page_type', 'unknown') if isinstance(page_data.get('content'), dict) else 'unknown'),
            'timestamp': page_data.get('scraped_at', '') or page_data.get('timestamp', ''),
            'headings': page_data.get('headings', []) or (page_data.get('content', {}).get('headings', []) if isinstance(page_data.get('content'), dict) else []),
            'lists': page_data.get('lists', []) or (page_data.get('content', {}).get('lists', []) if isinstance(page_data.get('content'), dict) else []),
            'domain': metadata_dict.get('domain', ''),
            'language': metadata_dict.get('language', ''),
            'content_length': page_data.get('content_length', 0)
        }
        
        # Create chunks
        chunks = self.create_chunks(
            text=clean_text,
            source_url=url,
            metadata=metadata
        )
        
        # Validate chunks
        valid_chunks = []
        for chunk in chunks:
            quality = self.validate_chunk_quality(chunk)
            if quality['is_valid'] and quality['quality_score'] >= 0.3:  # More lenient threshold
                chunk.metadata['quality_metrics'] = quality
                valid_chunks.append(chunk)
            else:
                logger.debug(f"Filtered out low-quality chunk: {quality['issues']}")
        
        logger.info(f"Processed page {url or 'unknown'}: "
                   f"{len(chunks)} chunks created, {len(valid_chunks)} passed quality check")
        
        return valid_chunks 