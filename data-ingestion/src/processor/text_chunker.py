"""
Text chunking strategy for breaking content into semantic chunks
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using simple sentence splitting")

from config.settings import get_settings


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    char_count: int
    heading: Optional[str] = None
    heading_level: Optional[int] = None


class TextChunker:
    """Intelligent text chunking with semantic preservation"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None, min_chunk_length: int = None):
        """Initialize the text chunker"""
        settings = get_settings()
        
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.min_chunk_length = min_chunk_length or settings.min_chunk_length
        
        logger.info(f"TextChunker initialized with chunk_size={self.chunk_size}")
    
    def _sentence_split(self, text: str) -> List[str]:
        """Split text into sentences"""
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Simple fallback
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """Chunk text by paragraphs, combining small ones"""
        if not text:
            return []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_position = 0
        
        for paragraph in paragraphs:
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = TextChunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=char_position,
                        end_char=char_position + len(current_chunk),
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_position += len(current_chunk)
                
                current_chunk = paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_length:
            chunk = TextChunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=char_position,
                end_char=char_position + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk)
            )
            chunks.append(chunk)
        
        logger.success(f"Created {len(chunks)} chunks by paragraphs")
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """Chunk text by sentences when paragraphs are too large"""
        if not text:
            return []
        
        sentences = self._sentence_split(text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = TextChunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=char_position,
                        end_char=char_position + len(current_chunk),
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_position += len(current_chunk)
                
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_length:
            chunk = TextChunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=char_position,
                end_char=char_position + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk)
            )
            chunks.append(chunk)
        
        logger.success(f"Created {len(chunks)} chunks by sentences")
        return chunks
    
    def smart_chunking(self, text: str, headings: List[Dict[str, Any]] = None) -> List[TextChunk]:
        """Smart chunking that preserves semantic meaning"""
        if not text:
            return []
        
        # Try paragraph chunking first
        chunks = self.chunk_by_paragraphs(text)
        
        # If chunks are too large, use sentence chunking
        large_chunks = [chunk for chunk in chunks if chunk.char_count > self.chunk_size]
        
        if large_chunks:
            logger.info(f"Re-chunking {len(large_chunks)} large chunks using sentences")
            final_chunks = []
            
            for chunk in chunks:
                if chunk.char_count > self.chunk_size:
                    # Re-chunk this large chunk by sentences
                    sentence_chunks = self.chunk_by_sentences(chunk.text)
                    final_chunks.extend(sentence_chunks)
                else:
                    final_chunks.append(chunk)
            
            # Re-index chunks
            for i, chunk in enumerate(final_chunks):
                chunk.chunk_index = i
            
            chunks = final_chunks
        
        logger.success(f"Smart chunking created {len(chunks)} chunks")
        return chunks 